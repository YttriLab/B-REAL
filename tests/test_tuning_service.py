"""Tests for backend.services.tuning_service.

The sweep drives real training subprocesses, so every test here replaces
training_service's start/read/get with an in-memory fake: the point is the
sampling, config routing, latency gate and cancellation logic, not torch.
"""

import time

import pytest

from backend.routers.tuning import TrainingConfig, _base_to_config, _build_search_space
from backend.services import tuning_service
from backend.services.tuning_service import (
    _arch_key,
    _is_oom,
    _parse_stages,
    get_task,
    read_all_results,
    start_tuning,
    stop_tuning,
)

optuna = pytest.importorskip("optuna")


BASE = {
    "data_loading": {"label_paths": ["l.csv"], "frame_dirs": ["f"],
                     "frames_per_group": 3, "label_union": None, "binarize": None},
    "data": {"train_percent": 0.7, "val_percent": 0.15, "test_percent": 0.15,
             "shuffle": False},
    "model": {"num_frames": 3, "variant": "b0", "shift_fraction": 0.125,
              "unidirectional": False, "pretrained": True, "tsm_stages": [5, 6, 7],
              "dropout": 0.2, "lr": 1e-4, "weight_decay": 1e-3,
              "temporal_pool": "last"},
    "data_loader": {"batch_size": 64, "num_workers": 4, "image_size": 224},
    "max_epochs": 3,
    "callbacks": {"early_stopping_patience": 20},
    "boundary_smoothing": {"enabled": False, "epsilon": 0.3, "window": 5},
}


class FakeTrainer:
    """Stands in for training_service: every run completes instantly with a
    fixed metric curve, and records the config it was handed."""

    def __init__(self, mcc_by_call=None, fail_with=None):
        self.configs = []
        self.stopped = []
        self._tasks = {}
        self._mcc = mcc_by_call or [0.5]
        self._fail_with = fail_with

    def start_training(self, config):
        tid = f"train-{len(self.configs)}"
        self.configs.append(config)
        if self._fail_with:
            self._tasks[tid] = {"status": "error", "error": self._fail_with,
                                "metrics": [], "latest_metrics": {}}
            return tid
        mcc = self._mcc[min(len(self.configs) - 1, len(self._mcc) - 1)]
        metrics = [
            {"status": "training", "epoch": e, "val_mcc": mcc, "val_loss": 1 - mcc}
            for e in range(3)
        ]
        self._tasks[tid] = {"status": "completed", "metrics": metrics,
                            "latest_metrics": {"best_checkpoint": f"{tid}.ckpt"}}
        return tid

    def read_all_metrics(self, tid):
        return self._tasks.get(tid, {}).get("metrics", [])

    def get_task(self, tid):
        return self._tasks.get(tid)

    def stop_training(self, tid):
        self.stopped.append(tid)
        t = self._tasks.get(tid)
        if t:
            t["status"] = "cancelled"
        return True


@pytest.fixture
def fake_trainer(monkeypatch):
    """Patch the names tuning_service imported, not the training_service module."""
    def _install(trainer):
        monkeypatch.setattr(tuning_service, "start_training", trainer.start_training)
        monkeypatch.setattr(tuning_service, "read_all_metrics", trainer.read_all_metrics)
        monkeypatch.setattr(tuning_service, "get_training_task", trainer.get_task)
        monkeypatch.setattr(tuning_service, "stop_training", trainer.stop_training)
        return trainer
    return _install


def run_sweep(**kwargs) -> str:
    """Start a sweep and wait for it to settle."""
    task_id = start_tuning(base_config=BASE, sampler_seed=0, **kwargs)
    for _ in range(200):
        if get_task(task_id)["status"] in ("completed", "error", "cancelled"):
            break
        time.sleep(0.05)
    return task_id


def events(task_id, kind):
    return [e for e in read_all_results(task_id) if e.get("event") == kind]


# ── helpers ────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_stages(self):
        assert _parse_stages("5,6,7") == [5, 6, 7]
        assert _parse_stages(" 1 , 2 ,x, 3 ") == [1, 2, 3]
        assert _parse_stages([5, 6]) == [5, 6]

    def test_is_oom(self):
        assert _is_oom("torch.cuda.OutOfMemoryError: CUDA out of memory")
        assert not _is_oom("FileNotFoundError: labels.csv")

    def test_arch_key_ignores_non_architecture_params(self):
        a = {"model": dict(BASE["model"]), "data_loader": dict(BASE["data_loader"])}
        b = {"model": dict(BASE["model"], lr=0.5, dropout=0.9),
             "data_loader": dict(BASE["data_loader"], batch_size=8)}
        assert _arch_key(a) == _arch_key(b)
        c = {"model": dict(BASE["model"], variant="b7"),
             "data_loader": dict(BASE["data_loader"])}
        assert _arch_key(a) != _arch_key(c)


# ── config marshalling ─────────────────────────────────────────────────────

class TestBaseToConfig:
    def test_carries_unidirectional_and_data_loader(self):
        cfg = _base_to_config(TrainingConfig(
            label_paths=["l"], frame_dirs=["f"], unidirectional=True,
            batch_size=16, image_size=380))
        # unidirectional used to be dropped here — sweeps silently trained
        # non-causal models that couldn't be deployed online.
        assert cfg["model"]["unidirectional"] is True
        assert cfg["data_loader"] == {"batch_size": 16, "num_workers": 4,
                                      "image_size": 380}


# ── sampling and routing ───────────────────────────────────────────────────

class TestSampling:
    def test_float_params_land_in_model(self, fake_trainer):
        t = fake_trainer(FakeTrainer())
        run_sweep(search_space={"lr": {"kind": "float", "low": 1e-5,
                                       "high": 1e-2, "log": True}},
                  n_trials=2)
        assert len(t.configs) == 2
        for cfg in t.configs:
            assert 1e-5 <= cfg["model"]["lr"] <= 1e-2
            assert cfg["model"]["lr"] != BASE["model"]["lr"]

    def test_categorical_variant_and_batch_size(self, fake_trainer):
        t = fake_trainer(FakeTrainer())
        run_sweep(search_space={
            "variant": {"kind": "categorical", "choices": ["b0", "b4"]},
            "batch_size": {"kind": "categorical", "choices": [8, 16]},
        }, n_trials=4)
        for cfg in t.configs:
            assert cfg["model"]["variant"] in ("b0", "b4")
            # batch_size must not leak into model — TSMClassifier would reject it
            assert "batch_size" not in cfg["model"]
            assert cfg["data_loader"]["batch_size"] in (8, 16)

    def test_tsm_stages_parsed_back_to_ints(self, fake_trainer):
        t = fake_trainer(FakeTrainer())
        tid = run_sweep(search_space={
            "tsm_stages": {"kind": "categorical", "choices": ["5,6,7", "1,2,3,4,5,6,7"]},
        }, n_trials=3)
        for cfg in t.configs:
            assert cfg["model"]["tsm_stages"] in ([5, 6, 7], [1, 2, 3, 4, 5, 6, 7])
        # the leaderboard still shows the readable string form
        assert all(isinstance(e["params"]["tsm_stages"], str)
                   for e in events(tid, "trial_start"))

    def test_frames_per_group_moves_data_and_model_together(self, fake_trainer):
        t = fake_trainer(FakeTrainer())
        run_sweep(search_space={
            "frames_per_group": {"kind": "categorical", "choices": [2, 4]},
        }, n_trials=4)
        for cfg in t.configs:
            # a mismatch here trains a model whose temporal window disagrees
            # with the clips it is fed
            assert cfg["data_loading"]["frames_per_group"] == cfg["model"]["num_frames"]
            assert cfg["model"]["num_frames"] in (2, 4)

    def test_wandb_disabled_per_trial(self, fake_trainer):
        t = fake_trainer(FakeTrainer())
        run_sweep(search_space={"dropout": {"kind": "float", "low": 0.0, "high": 0.6}},
                  n_trials=1)
        assert t.configs[0]["use_wandb"] is False


# ── objective / results ────────────────────────────────────────────────────

class TestResults:
    def test_reports_best_trial(self, fake_trainer):
        fake_trainer(FakeTrainer(mcc_by_call=[0.1, 0.9, 0.4]))
        tid = run_sweep(search_space={"lr": {"kind": "float", "low": 1e-5,
                                             "high": 1e-2, "log": True}},
                        n_trials=3)
        done = events(tid, "done")[0]
        assert done["best_trial"] == 1
        assert done["best_value"] == pytest.approx(0.9)

    def test_val_loss_is_minimized(self, fake_trainer):
        fake_trainer(FakeTrainer(mcc_by_call=[0.1, 0.9]))  # val_loss = 0.9, 0.1
        tid = run_sweep(search_space={"lr": {"kind": "float", "low": 1e-5,
                                             "high": 1e-2, "log": True}},
                        n_trials=2, objective_metric="val_loss")
        assert events(tid, "done")[0]["best_value"] == pytest.approx(0.1)

    def test_oom_is_reported_separately_from_a_generic_failure(self, fake_trainer):
        fake_trainer(FakeTrainer(fail_with="torch.cuda.OutOfMemoryError: CUDA out of memory"))
        tid = run_sweep(search_space={
            "variant": {"kind": "categorical", "choices": ["b6", "b7"]},
        }, n_trials=2)
        failed = events(tid, "trial_failed")
        assert failed and all(e["reason"] == "oom" for e in failed)
        # a failed trial doesn't abort the sweep
        assert get_task(tid)["status"] == "completed"


# ── latency gate ───────────────────────────────────────────────────────────

class TestLatencyGate:
    def _bench(self, monkeypatch, p99_by_variant, calls=None):
        def fake_measure(cfg, latency):
            variant = cfg["model"]["variant"]
            if calls is not None:
                calls.append(_arch_key(cfg))
            return {"ms_per_batch_p99": p99_by_variant[variant],
                    "device_name": "FakeGPU", "precision": latency["precision"]}
        monkeypatch.setattr(tuning_service, "_measure_latency", fake_measure)

    def test_rejects_architectures_over_budget(self, monkeypatch, fake_trainer):
        t = fake_trainer(FakeTrainer())
        self._bench(monkeypatch, {"b0": 10.0, "b7": 120.0})
        tid = run_sweep(search_space={
            "variant": {"kind": "categorical", "choices": ["b0", "b7"]},
        }, n_trials=6, latency={"max_p99_ms": 33.0, "batch_size": 1,
                                "precision": "fp32", "iters": 5, "warmup": 1})

        rejected = events(tid, "trial_rejected")
        assert rejected, "expected the slow variant to be rejected"
        assert all(e["params"]["variant"] == "b7" for e in rejected)
        assert all(e["p99_ms"] == 120.0 for e in rejected)
        # rejected trials never reach training
        assert all(c["model"]["variant"] == "b0" for c in t.configs)
        assert len(t.configs) + len(rejected) == 6

    def test_benchmark_is_cached_per_architecture(self, monkeypatch, fake_trainer):
        fake_trainer(FakeTrainer())
        calls = []
        self._bench(monkeypatch, {"b0": 10.0}, calls=calls)
        run_sweep(search_space={"lr": {"kind": "float", "low": 1e-5, "high": 1e-2,
                                       "log": True}},
                  n_trials=5, latency={"max_p99_ms": 33.0, "batch_size": 1,
                                       "precision": "fp32", "iters": 5, "warmup": 1})
        # the architecture never varies, so it is benchmarked exactly once
        assert len(calls) == 1

    def test_unmeasurable_budget_aborts_the_sweep(self, monkeypatch, fake_trainer):
        """A constraint that can't be measured is not a constraint. Rather than
        train models that might miss the frame budget, the sweep stops."""
        t = fake_trainer(FakeTrainer())

        def boom(cfg, latency):
            raise RuntimeError("tensorrt is not installed")
        monkeypatch.setattr(tuning_service, "_measure_latency", boom)

        tid = run_sweep(search_space={"lr": {"kind": "float", "low": 1e-5,
                                             "high": 1e-2, "log": True}},
                        n_trials=5, latency={"max_p99_ms": 33.0, "batch_size": 1,
                                             "precision": "fp32", "iters": 5,
                                             "warmup": 1})
        assert get_task(tid)["status"] == "error"
        assert "tensorrt is not installed" in get_task(tid)["error"]
        assert t.configs == []  # nothing trained
        err = events(tid, "error")
        assert err and err[0]["reason"] == "latency"

    def test_target_is_passed_through_to_the_benchmark(self, monkeypatch, fake_trainer):
        fake_trainer(FakeTrainer())
        seen = {}

        def capture(cfg, latency):
            seen.update(latency)
            return {"ms_per_batch_p99": 5.0, "device_name": "CPU",
                    "precision": "fp32", "runtime": "openvino"}
        monkeypatch.setattr(tuning_service, "_measure_latency", capture)

        tid = run_sweep(search_space={"lr": {"kind": "float", "low": 1e-5,
                                             "high": 1e-2, "log": True}},
                        n_trials=1, latency={"max_p99_ms": 33.0, "batch_size": 1,
                                             "precision": "fp32", "iters": 5,
                                             "warmup": 1, "target": "openvino"})
        assert seen["target"] == "openvino"
        # the runtime the number came from is reported, so a leaderboard can't
        # be misread as TorchScript timings
        assert events(tid, "latency")[0]["runtime"] == "openvino"


# ── cancellation ───────────────────────────────────────────────────────────

class TestCancellation:
    def test_stop_marks_the_sweep_cancelled(self, fake_trainer):
        class SlowTrainer(FakeTrainer):
            def get_task(self, tid):
                time.sleep(0.2)  # keep the trial in-flight long enough to cancel
                return super().get_task(tid)

        fake_trainer(SlowTrainer())
        task_id = start_tuning(base_config=BASE, sampler_seed=0, n_trials=10,
                               search_space={"lr": {"kind": "float", "low": 1e-5,
                                                    "high": 1e-2, "log": True}})
        time.sleep(0.3)
        assert stop_tuning(task_id) is True
        for _ in range(100):
            if get_task(task_id)["status"] in ("cancelled", "completed", "error"):
                break
            time.sleep(0.05)
        assert get_task(task_id)["status"] == "cancelled"

    def test_stop_on_a_finished_sweep_returns_false(self, fake_trainer):
        fake_trainer(FakeTrainer())
        tid = run_sweep(search_space={"dropout": {"kind": "float", "low": 0.0,
                                                  "high": 0.6}}, n_trials=1)
        assert stop_tuning(tid) is False
        assert stop_tuning("nope") is False


# ── request validation (router-level helper) ───────────────────────────────

class TestSearchSpaceBuilding:
    def _req(self, **kw):
        from backend.routers.tuning import TuningRequest
        return TuningRequest(base=TrainingConfig(label_paths=["l"], frame_dirs=["f"]),
                             **kw)

    def test_tags_kinds(self):
        space = _build_search_space(self._req(
            lr={"low": 1e-5, "high": 1e-2, "log": True},
            variant={"choices": ["b0", "b3"]}))
        assert space["lr"]["kind"] == "float"
        assert space["variant"] == {"kind": "categorical", "choices": ["b0", "b3"]}

    def test_empty_when_nothing_selected(self):
        assert _build_search_space(self._req()) == {}

    def test_rejects_an_empty_choice_list(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as e:
            _build_search_space(self._req(variant={"choices": []}))
        assert e.value.status_code == 422
