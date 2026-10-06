"""
Hyperparameter tuning service: runs an Optuna study that sweeps training
hyperparameters, one trial at a time, by reusing the normal training subprocess.

Each trial:
  * samples hyperparameters from the search space,
  * launches a training run via training_service.start_training,
  * reports the per-epoch objective metric (default val_mcc) to Optuna so bad
    trials can be pruned early,
  * records the trial's best objective value + checkpoint.

Results are written as JSON-lines events to a file tailed by the SSE endpoint,
mirroring the training service pattern. Optuna is imported lazily so the rest of
the backend works even if it isn't installed.
"""

import json
import math
import os
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from typing import Optional

from backend.services.training_service import (
    start_training,
    stop_training,
    read_all_metrics,
    get_task as get_training_task,
)

# ── Task registry ──────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancelled"] = True
        ttid = task.get("active_train_task")
        if ttid:
            try:
                stop_training(ttid)
            except Exception:
                pass


class _SweepCancelled(Exception):
    """Raised inside the objective to abort the whole study (not just one trial)."""


class _LatencyUnmeasurable(Exception):
    """The frame budget couldn't be measured (runtime missing, export failed).

    Aborts the study: without a measurement the constraint isn't being enforced,
    and a sweep that silently ignores it produces models that can't ship.
    """


def _emit(results_path: str, payload: dict):
    """Append one JSON-lines event to the results file (tailed by the SSE endpoint)."""
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


# Metric → optimization direction. val_loss is minimized; the accuracy-like
# metrics are maximized.
_DIRECTION = {
    "val_mcc": "maximize",
    "val_balanced_acc": "maximize",
    "val_acc": "maximize",
    "val_loss": "minimize",
}


def _objective_value(epochs: list[dict], metric: str, direction: str):
    """Best objective value seen across a trial's epochs (max or min by direction)."""
    vals = [e[metric] for e in epochs if metric in e and e[metric] is not None]
    if not vals:
        return None
    return max(vals) if direction == "maximize" else min(vals)


# Where a sampled hyperparameter lands in the config_tsm.yaml-shaped dict.
# frames_per_group is coupled: the data window and the model's temporal window
# have to agree, so it writes both places.
_SECTION = {
    "lr": "model",
    "weight_decay": "model",
    "dropout": "model",
    "variant": "model",
    "temporal_pool": "model",
    "tsm_stages": "model",
    "batch_size": "data_loader",
    "image_size": "data_loader",
}

# Architecture-shaped params — the only ones that change inference latency.
_ARCH_PARAMS = ("variant", "tsm_stages", "temporal_pool", "num_frames", "image_size")


def _parse_stages(text) -> list:
    """'5,6,7' -> [5,6,7]. Optuna categoricals must be hashable, so stage lists
    travel as comma-strings and are parsed back after sampling."""
    if isinstance(text, (list, tuple)):
        return [int(v) for v in text]
    out = []
    for part in str(text).replace(" ", "").split(","):
        if part:
            try:
                out.append(int(part))
            except ValueError:
                pass
    return out


def _sample_params(trial, search_space: dict, cfg: dict) -> dict:
    """Sample every hyperparameter in the space and write it into `cfg`.

    Returns the sampled values in their reportable (pre-parse) form, for the
    leaderboard and for Optuna's own record.
    """
    params = {}
    for name, spec in search_space.items():
        kind = spec.get("kind", "float")
        if kind == "float":
            value = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        else:
            value = trial.suggest_categorical(name, spec["choices"])
        params[name] = value

        if name == "frames_per_group":
            cfg.setdefault("data_loading", {})["frames_per_group"] = int(value)
            cfg.setdefault("model", {})["num_frames"] = int(value)
            continue

        section = cfg.setdefault(_SECTION[name], {})
        section[name] = _parse_stages(value) if name == "tsm_stages" else value

    return params


def _arch_key(cfg: dict) -> tuple:
    """Identity of the architecture being benchmarked — latency depends on this
    and nothing else, so results are cached against it."""
    model = cfg.get("model", {})
    dl = cfg.get("data_loader", {})
    return (
        model.get("variant", "b0"),
        tuple(model.get("tsm_stages") or [5, 6, 7]),
        model.get("temporal_pool", "last"),
        int(model.get("num_frames", 3)),
        int(dl.get("image_size", 224)),
        bool(model.get("unidirectional", False)),
    )


def _measure_latency(cfg: dict, latency: dict) -> dict:
    """Benchmark the sampled architecture in its deployment format.

    OpenVINO for a CPU rig, TensorRT for a GPU one — not TorchScript, which is
    fast to produce but isn't what runs in production and mispredicts the frame
    budget. Imported lazily (it pulls in torch) so the module stays importable in
    environments that only serve the API surface.
    """
    from optimized_runtime import benchmark_optimized

    model = cfg.get("model", {})
    dl = cfg.get("data_loader", {})
    return benchmark_optimized(
        target=latency.get("target", "auto"),
        variant=model.get("variant", "b0"),
        num_frames=int(model.get("num_frames", 3)),
        tsm_stages=model.get("tsm_stages"),
        temporal_pool=model.get("temporal_pool", "last"),
        unidirectional=bool(model.get("unidirectional", False)),
        image_size=int(dl.get("image_size", 224)),
        batch_size=int(latency.get("batch_size", 1)),
        precision=latency.get("precision", "fp32"),
        iters=int(latency.get("iters", 100)),
        warmup=int(latency.get("warmup", 20)),
    )


def _is_oom(text: str) -> bool:
    """Distinguish 'this architecture doesn't fit in VRAM' from a real failure —
    the big EfficientNet variants at large batches hit this routinely, and the
    leaderboard should say so rather than showing a generic error."""
    low = (text or "").lower()
    return "out of memory" in low or "outofmemoryerror" in low


def start_tuning(
    base_config: dict,
    search_space: dict,
    n_trials: int = 20,
    objective_metric: str = "val_mcc",
    sampler_seed: Optional[int] = None,
    latency: Optional[dict] = None,
) -> str:
    """
    Launch an Optuna study in a background thread.

    Args:
        base_config: a full training config dict (same shape training_service
            consumes) used as the fixed baseline; sampled hyperparameters
            override entries under config["model"].
        search_space: which hyperparameters to sweep, each tagged by kind, e.g.
            {
              "lr":      {"kind": "float", "low": 1e-5, "high": 1e-2, "log": True},
              "dropout": {"kind": "float", "low": 0.0,  "high": 0.6},
              "variant": {"kind": "categorical", "choices": ["b0", "b1", "b4"]},
            }
        n_trials: number of trials to run.
        objective_metric: per-epoch metric to optimize (val_mcc/val_loss/...).
        sampler_seed: optional seed for reproducible sampling (mainly for tests).
        latency: optional constraint, e.g. {"max_p99_ms": 25.0, "batch_size": 1,
            "precision": "fp16"}. Architectures whose benchmarked p99 exceeds the
            budget are rejected before training — the sweep only spends GPU time
            on models that could actually be deployed.

    Returns:
        task_id for status polling.
    """
    direction = _DIRECTION.get(objective_metric, "maximize")

    task_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix="breal_tune_")
    results_path = os.path.join(tmp_dir, "results.jsonl")
    open(results_path, "w", encoding="utf-8").close()

    task = {
        "status": "starting",
        "results_path": results_path,
        "tmp_dir": tmp_dir,
        "cancelled": False,
        "active_train_task": None,
        "best": None,
        "error": None,
        "n_trials": n_trials,
        "objective_metric": objective_metric,
        "direction": direction,
        "latency": latency,
        "latency_cache": {},   # arch key -> benchmark result (or None if it failed)
    }
    _tasks[task_id] = task

    def _make_objective(optuna_mod):
        def objective(trial):
            if task["cancelled"]:
                raise _SweepCancelled()

            cfg = deepcopy(base_config)
            params = _sample_params(trial, search_space, cfg)

            # Per-trial training runs don't each get their own W&B run — the sweep
            # leaderboard is the experiment record.
            cfg["use_wandb"] = False
            cfg.pop("wandb_api_key", None)

            _emit(results_path, {"event": "trial_start", "trial": trial.number, "params": params})

            # Latency gate: reject architectures that can't hit the frame budget
            # before spending a training run on them. Cached per architecture, so
            # a sweep over lr with a fixed architecture benchmarks exactly once.
            if latency:
                key = _arch_key(cfg)
                cache = task["latency_cache"]
                if key not in cache:
                    try:
                        cache[key] = _measure_latency(cfg, latency)
                    except Exception as e:
                        # A constraint that can't be measured is not a constraint.
                        # Abort rather than quietly training models that may not
                        # be deployable — the user asked for this gate.
                        raise _LatencyUnmeasurable(
                            f"{type(e).__name__}: {e}"[-500:]) from e
                bench = cache[key]
                p99 = bench.get("ms_per_batch_p99")
                budget = latency["max_p99_ms"]
                _emit(results_path, {
                    "event": "latency", "trial": trial.number,
                    "p99_ms": p99, "budget_ms": budget,
                    "device": bench.get("device_name"),
                    "precision": bench.get("precision"),
                    "runtime": bench.get("runtime"),
                })
                if p99 is not None and p99 > budget:
                    _emit(results_path, {
                        "event": "trial_rejected", "trial": trial.number,
                        "reason": "latency", "p99_ms": p99, "budget_ms": budget,
                        "runtime": bench.get("runtime"), "params": params,
                    })
                    raise optuna_mod.TrialPruned()

            ttid = start_training(cfg)
            task["active_train_task"] = ttid
            seen = 0

            while True:
                if task["cancelled"]:
                    stop_training(ttid)
                    raise _SweepCancelled()

                epochs = [
                    e for e in read_all_metrics(ttid)
                    if e.get("status") == "training" and "epoch" in e
                ]
                for e in epochs[seen:]:
                    seen += 1
                    val = e.get(objective_metric)
                    if val is None:
                        continue
                    _emit(results_path, {
                        "event": "epoch", "trial": trial.number,
                        "epoch": e["epoch"], objective_metric: val,
                    })
                    # Optuna prunes on a maximization basis, so flip sign when we
                    # are minimizing (val_loss).
                    trial.report(val if direction == "maximize" else -val, e["epoch"])
                    if trial.should_prune():
                        stop_training(ttid)
                        best = _objective_value(epochs[:seen], objective_metric, direction)
                        _emit(results_path, {
                            "event": "trial_pruned", "trial": trial.number,
                            "value": best, "params": params,
                        })
                        raise optuna_mod.TrialPruned()

                t = get_training_task(ttid)
                status = t.get("status") if t else "error"
                if status in ("completed", "error", "cancelled"):
                    break
                time.sleep(1.0)

            task["active_train_task"] = None

            if status == "error":
                t = get_training_task(ttid) or {}
                err = (t.get("error") or "training failed")
                _emit(results_path, {
                    "event": "trial_failed", "trial": trial.number,
                    "reason": "oom" if _is_oom(err) else "error",
                    "error": err[-500:], "params": params,
                })
                # Returning a worst-case value (instead of raising) keeps the study
                # going and lets Optuna learn to avoid this region.
                return -math.inf if direction == "maximize" else math.inf

            epochs = [
                e for e in read_all_metrics(ttid)
                if e.get("status") == "training" and "epoch" in e
            ]
            value = _objective_value(epochs, objective_metric, direction)
            if value is None:
                value = -math.inf if direction == "maximize" else math.inf

            final = (get_training_task(ttid) or {}).get("latest_metrics", {})
            _emit(results_path, {
                "event": "trial_done", "trial": trial.number,
                "value": value, "params": params,
                "checkpoint": final.get("best_checkpoint"),
            })
            return value

        return objective

    def _run():
        try:
            try:
                import optuna
                from optuna.samplers import TPESampler
                from optuna.pruners import MedianPruner
            except ImportError:
                task["status"] = "error"
                task["error"] = (
                    "Optuna is not installed. Run: pip install optuna "
                    "(into the env the backend uses)."
                )
                _emit(results_path, {"event": "error", "error": task["error"]})
                return

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            task["status"] = "running"
            _emit(results_path, {
                "event": "start", "n_trials": n_trials,
                "objective": objective_metric, "direction": direction,
            })

            study = optuna.create_study(
                direction=direction,
                sampler=TPESampler(seed=sampler_seed),
                pruner=MedianPruner(n_warmup_steps=2),
            )
            task["study"] = study

            # RuntimeError-free: failed trials return a worst value rather than raise,
            # so we don't need `catch`. _SweepCancelled propagates to abort the study.
            study.optimize(_make_objective(optuna), n_trials=n_trials)

            completed = [t for t in study.trials
                         if t.value is not None and math.isfinite(t.value)]
            if completed:
                best = study.best_trial
                task["best"] = {
                    "trial": best.number, "value": best.value, "params": best.params,
                }
                _emit(results_path, {
                    "event": "done", "best_trial": best.number,
                    "best_value": best.value, "best_params": best.params,
                })
            else:
                _emit(results_path, {"event": "done", "best_trial": None,
                                     "best_value": None, "best_params": None})
            task["status"] = "completed"
        except _SweepCancelled:
            task["status"] = "cancelled"
            _emit(results_path, {"event": "cancelled"})
        except _LatencyUnmeasurable as e:
            task["error"] = (
                f"Latency constraint could not be measured, so the sweep stopped "
                f"rather than train models that may miss the frame budget: {e}"
            )
            task["status"] = "error"
            _emit(results_path, {"event": "error", "reason": "latency",
                                 "error": task["error"]})
        except Exception as e:
            task["error"] = str(e) or repr(e)
            task["status"] = "error"
            _emit(results_path, {"event": "error", "error": task["error"]})

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def read_all_results(task_id: str) -> list[dict]:
    """Read every JSON-lines event written so far (for ordered SSE streaming)."""
    task = get_task(task_id)
    if not task:
        return []
    results_path = task.get("results_path")
    if not results_path or not os.path.exists(results_path):
        return []
    out = []
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def stop_tuning(task_id: str) -> bool:
    """Request cancellation of a running sweep."""
    task = get_task(task_id)
    if not task:
        return False
    if task["status"] in ("completed", "error", "cancelled"):
        return False
    task["cancelled"] = True
    ttid = task.get("active_train_task")
    if ttid:
        try:
            stop_training(ttid)
        except Exception:
            pass
    return True
