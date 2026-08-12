"""
TSM-EfficientNet model for temporal behavior classification.

Uses Temporal Shift Modules (TSM) inserted into EfficientNet MBConv blocks
to capture temporal information across consecutive frames without extra
computation. The channel-shifting mechanism allows standard 2D convolutions
to implicitly reason about temporal context.

Architecture:
    Input: (B, T, C, H, W) triplet of frames
    → Reshape to (B*T, C, H, W)
    → EfficientNet with TSM in each MBConv stage
    → Pool per-frame features → select last frame or temporal pool
    → Linear classifier → logits
"""

import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, MatthewsCorrCoef, ConfusionMatrix
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torchvision import models
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.tuner import Tuner
import os
import cv2
import wandb

# Reuse data loading, dataset, datamodule, and splitting from model.py
from model import load_data, MultiImageDataModule, train_test_split


# ---------------------------------------------------------------------------
# Temporal Shift Module
# ---------------------------------------------------------------------------

class TemporalShift(nn.Module):
    """
    Temporal Shift Module (Lin et al., 2019).

    Shifts a fraction of channels forward/backward along the temporal axis.
    Inserted before each residual block so that subsequent 2D convolutions
    implicitly mix information from adjacent frames.

    Args:
        num_frames: Number of frames in the temporal window (T).
        shift_fraction: Fraction of channels to shift in each direction.
            Total shifted = 2 * shift_fraction (half forward, half backward).
        unidirectional: If True, use a CAUSAL shift — all shifted channels come
            from PAST frames only (t-1), never t+1. This matches online/real-time
            inference where future frames don't exist yet, avoiding a train/deploy
            mismatch on a live rig. The same total channel count (2*shift_fraction)
            is shifted, so model capacity is comparable to the bidirectional shift.
    """

    def __init__(self, num_frames: int = 3, shift_fraction: float = 0.125,
                 unidirectional: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.shift_fraction = shift_fraction
        self.unidirectional = unidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        BT, C, H, W = x.shape
        T = self.num_frames
        B = BT // T

        x = x.view(B, T, C, H, W)

        shift_c = int(C * self.shift_fraction)

        out = x.clone()
        if self.unidirectional:
            # Causal: shift 2*shift_c channels from the PAST only (t-1). No future
            # leakage, so training matches online inference frame-for-frame.
            n = min(2 * shift_c, C)
            out[:, 1:, :n, :, :] = x[:, :-1, :n, :, :]
            out[:, 0, :n, :, :] = 0  # zero-pad first frame (no past)
        else:
            # Bidirectional (default): half from past, half from future.
            # Shift forward: frame t gets channels from frame t-1
            out[:, 1:, :shift_c, :, :] = x[:, :-1, :shift_c, :, :]
            out[:, 0, :shift_c, :, :] = 0  # zero-pad first frame

            # Shift backward: frame t gets channels from frame t+1
            out[:, :-1, shift_c:2*shift_c, :, :] = x[:, 1:, shift_c:2*shift_c, :, :]
            out[:, -1, shift_c:2*shift_c, :, :] = 0  # zero-pad last frame

        # Remaining channels unchanged (already from clone)

        return out.view(BT, C, H, W)


class TSMWrappedBlock(nn.Module):
    """Wraps an existing nn.Module (e.g. MBConv block) with a TSM shift."""

    def __init__(self, block: nn.Module, num_frames: int = 3, shift_fraction: float = 0.125,
                 unidirectional: bool = False):
        super().__init__()
        self.tsm = TemporalShift(num_frames=num_frames, shift_fraction=shift_fraction,
                                 unidirectional=unidirectional)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tsm(x)
        return self.block(x)


# ---------------------------------------------------------------------------
# TSM-EfficientNet backbone
# ---------------------------------------------------------------------------

def build_tsm_efficientnet(
    variant: str = "b0",
    num_frames: int = 3,
    shift_fraction: float = 0.125,
    pretrained: bool = True,
    tsm_stages: list = None,
    unidirectional: bool = False,
):
    """
    Build an EfficientNet with TSM inserted into selected MBConv stages.

    EfficientNet structure (torchvision):
        features[0]   — stem Conv2dNormActivation
        features[1-7] — MBConv stages (each is a Sequential of InvertedResidual blocks)
        features[8]   — head Conv2dNormActivation
        avgpool        — AdaptiveAvgPool2d
        classifier     — Sequential(Dropout, Linear)

    TSM is inserted before each InvertedResidual block in the selected stages.
    All stages before the first TSM stage are frozen (including stem).

    Args:
        variant: EfficientNet variant ("b0", "b1", "b2", etc.)
        num_frames: Temporal window size.
        shift_fraction: Fraction of channels shifted per direction.
        pretrained: Use ImageNet pretrained weights.
        tsm_stages: List of stage indices (1-7) to insert TSM into.
            Stages before the first TSM stage are frozen.
            Defaults to [5, 6, 7] (last 3 MBConv stages).

    Returns:
        (backbone, feature_dim): The modified EfficientNet and its output feature dim.
    """
    if tsm_stages is None:
        tsm_stages = [5, 6, 7]

    weights = "DEFAULT" if pretrained else None
    print(f"[build_tsm_efficientnet] pretrained={pretrained}, weights={weights}")

    # b4-b7 share b0-b3's layout (features[0] stem, features[1-7] MBConv,
    # classifier[1]), so everything below works unchanged. They are ImageNet-trained
    # at 380/456/528/600px though, so at the data module's default 224 they cost
    # their full parameter count for little gain — raise image_size with them.
    builder = {
        "b0": models.efficientnet_b0,
        "b1": models.efficientnet_b1,
        "b2": models.efficientnet_b2,
        "b3": models.efficientnet_b3,
        "b4": models.efficientnet_b4,
        "b5": models.efficientnet_b5,
        "b6": models.efficientnet_b6,
        "b7": models.efficientnet_b7,
    }
    if variant not in builder:
        raise ValueError(f"Unsupported EfficientNet variant '{variant}'. Choose from {list(builder.keys())}")

    net = builder[variant](weights=weights)

    # Verify initialization
    sample_weight = net.features[1][0].block[0][0].weight
    print(f"[build_tsm_efficientnet] Sample weight stats: mean={sample_weight.mean().item():.6f}, std={sample_weight.std().item():.6f}")

    # Get feature dim before we modify the classifier
    feature_dim = net.classifier[1].in_features  # 1280 for b0

    # Remove original classifier — we'll add our own
    net.classifier = nn.Identity()

    first_tsm_stage = min(tsm_stages)

    # Freeze everything before the first TSM stage (stem + early MBConv stages)
    for stage_idx in range(0, first_tsm_stage):
        for param in net.features[stage_idx].parameters():
            param.requires_grad = False

    # Wrap InvertedResidual blocks with TSM only in selected stages
    for stage_idx in range(1, 8):  # stages 1-7
        if stage_idx not in tsm_stages:
            continue
        stage = net.features[stage_idx]
        wrapped_blocks = []
        for block in stage:
            wrapped_blocks.append(
                TSMWrappedBlock(block, num_frames=num_frames, shift_fraction=shift_fraction,
                                unidirectional=unidirectional)
            )
        net.features[stage_idx] = nn.Sequential(*wrapped_blocks)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"TSM stages: {tsm_stages} | Frozen before stage {first_tsm_stage}")
    print(f"Backbone params: {trainable:,} trainable, {frozen:,} frozen")

    return net, feature_dim


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class TSMClassifier(L.LightningModule):
    """
    TSM-EfficientNet classifier for temporal behavior classification.

    Input: (B, T, C, H, W) — batch of frame triplets.
    Output: (B, num_classes) logits.
    """

    def __init__(
        self,
        num_frames: int = 3,
        variant: str = "b0",
        shift_fraction: float = 0.125,
        pretrained: bool = True,
        tsm_stages: list = None,
        unidirectional: bool = False,  # causal shift (past-only) for online inference
        freeze_stem: bool = True,  # backward compat (ignored, use tsm_stages)
        dropout: float = 0.2,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        max_epochs: int = 100,
        temporal_pool: str = "last",  # "last", "avg", or "max"
        weights=None,
        class_names=None,
        boundary_smoothing=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["weights", "freeze_stem"])

        if class_names is not None:
            self.num_classes = len(class_names)
            self.class_names = class_names
        else:
            # Checkpoint reload path: class_names not passed, infer from head weights
            self.num_classes = 2  # will be overwritten by state_dict load
            self.class_names = [f"Class_{i}" for i in range(self.num_classes)]
        self.num_frames = num_frames

        task = "binary" if self.num_classes <= 2 else "multiclass"

        # Build backbone
        self.backbone, feature_dim = build_tsm_efficientnet(
            variant=variant,
            num_frames=num_frames,
            shift_fraction=shift_fraction,
            pretrained=pretrained,
            tsm_stages=tsm_stages,
            unidirectional=unidirectional,
        )

        # Classifier head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, self.num_classes),
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Boundary-aware label smoothing
        bs = boundary_smoothing or {}
        self.boundary_smoothing_enabled = bs.get("enabled", False)
        self.boundary_smoothing_epsilon = bs.get("epsilon", 0.1)
        self.boundary_smoothing_window = bs.get("window", 5)
        self.register_buffer("class_weights", weights if weights is not None else torch.ones(self.num_classes))

        # Metrics
        self.train_acc = Accuracy(task=task, num_classes=self.num_classes, average="macro")
        self.val_acc = Accuracy(task=task, num_classes=self.num_classes, average="macro")
        self.test_acc = Accuracy(task=task, num_classes=self.num_classes, average="macro")

        self.train_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.val_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.test_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)

        self.test_confusion = ConfusionMatrix(task=task, num_classes=self.num_classes)

        self.test_preds = []
        self.test_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        assert T == self.num_frames, f"Expected {self.num_frames} frames, got {T}"

        # Reshape for 2D backbone: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Run through TSM-EfficientNet features + avgpool
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # (B*T, feature_dim)

        # Reshape back to (B, T, feature_dim)
        x = x.view(B, T, -1)

        # Temporal pooling
        if self.hparams.temporal_pool == "last":
            x = x[:, -1, :]  # (B, feature_dim)
        elif self.hparams.temporal_pool == "avg":
            x = x.mean(dim=1)
        elif self.hparams.temporal_pool == "max":
            x = x.max(dim=1).values
        else:
            raise ValueError(f"Unknown temporal_pool: {self.hparams.temporal_pool}")

        logits = self.head(x)
        return logits

    # --- Boundary-aware label smoothing (same as model.py) ---

    def _boundary_smoothed_loss(self, logits, labels, boundary_dists):
        eps = self.boundary_smoothing_epsilon
        window = self.boundary_smoothing_window

        per_epsilon = eps * (1.0 - boundary_dists.float() / window).clamp(min=0.0)

        one_hot = F.one_hot(labels, self.num_classes).float()
        soft_targets = one_hot * (1.0 - per_epsilon.unsqueeze(1)) + per_epsilon.unsqueeze(1) / self.num_classes

        log_probs = F.log_softmax(logits, dim=1)
        per_sample_loss = -(soft_targets * log_probs).sum(dim=1)

        sample_weights = self.class_weights[labels]
        loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()

        return loss, per_epsilon

    # --- Training / validation / test steps ---

    def training_step(self, batch, batch_idx):
        images, labels, boundary_dists = batch
        logits = self(images)

        if self.boundary_smoothing_enabled:
            loss, per_epsilon = self._boundary_smoothed_loss(logits, labels, boundary_dists)
            self.log("train_mean_epsilon", per_epsilon.mean(), on_step=False, on_epoch=True)
            self.log("train_smoothed_frac", (per_epsilon > 0).float().mean(), on_step=False, on_epoch=True)
        else:
            loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_balanced_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        # Log the Metric object, rather than its per-batch return tensor, so
        # Lightning calls compute() once over the complete epoch state. MCC is
        # nonlinear: averaging batch MCC values is not the dataset-wide MCC.
        self.train_mcc.update(preds, labels)
        self.log("train_mcc", self.train_mcc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, boundary_dists = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_balanced_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.val_mcc.update(preds, labels)
        self.log("val_mcc", self.val_mcc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels, boundary_dists = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.log("test_loss", loss)
        self.log("test_balanced_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True)
        self.test_mcc.update(preds, labels)
        self.log("test_mcc", self.test_mcc, on_step=False, on_epoch=True)

        self.test_confusion.update(preds, labels)
        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        cm = self.test_confusion.compute().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_normalized, vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Fraction of True")
        ax.set_xticks(range(self.num_classes), labels=self.class_names, rotation=45, ha="right")
        ax.set_yticks(range(self.num_classes), labels=self.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Test Confusion Matrix")
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        self.logger.experiment.log({"Confusion Matrix - Test Set": wandb.Image(fig)})
        plt.close(fig)
        self.test_confusion.reset()

        # Realtime metrics (2-of-3 majority rule, per clip)
        if hasattr(self, "test_clip_sizes") and self.test_clip_sizes is not None:
            all_preds = torch.cat(self.test_preds).numpy()
            all_labels = torch.cat(self.test_labels).numpy()
            positive_class = 1
            transition_margin = 3

            clip_ious_raw, clip_ious_smoothed = [], []
            clip_hit_rates, clip_fprs = [], []

            offset = 0
            for clip_idx, clip_size in enumerate(self.test_clip_sizes):
                clip_preds = all_preds[offset:offset + clip_size]
                clip_labels = all_labels[offset:offset + clip_size]
                offset += clip_size

                gt_pos = clip_labels == positive_class

                # Raw IoU
                pred_pos_raw = clip_preds == positive_class
                inter_raw = np.sum(gt_pos & pred_pos_raw)
                union_raw = np.sum(gt_pos | pred_pos_raw)
                iou_raw = inter_raw / union_raw if union_raw > 0 else float("nan")
                clip_ious_raw.append(iou_raw)

                # 2-of-3 majority rule
                smoothed_preds = np.zeros_like(clip_preds)
                for i in range(len(clip_preds)):
                    start = max(0, i - 2)
                    window = clip_preds[start:i + 1]
                    smoothed_preds[i] = positive_class if np.sum(window == positive_class) >= 2 else 0

                pred_pos_sm = smoothed_preds == positive_class
                inter_sm = np.sum(gt_pos & pred_pos_sm)
                union_sm = np.sum(gt_pos | pred_pos_sm)
                iou_sm = inter_sm / union_sm if union_sm > 0 else float("nan")
                clip_ious_smoothed.append(iou_sm)

                # Hit rate
                bout_starts, bout_ends = [], []
                in_bout = False
                for i in range(len(clip_labels)):
                    if clip_labels[i] == positive_class and not in_bout:
                        bout_starts.append(i)
                        in_bout = True
                    elif clip_labels[i] != positive_class and in_bout:
                        bout_ends.append(i)
                        in_bout = False
                if in_bout:
                    bout_ends.append(len(clip_labels))

                total_bouts = len(bout_starts)
                if total_bouts > 0:
                    bouts_hit = sum(
                        1 for bs, be in zip(bout_starts, bout_ends)
                        if np.any(smoothed_preds[bs:be] == positive_class)
                    )
                    hit_rate = bouts_hit / total_bouts
                else:
                    hit_rate = float("nan")
                clip_hit_rates.append(hit_rate)

                # FPR
                transition_indices = []
                for i in range(1, len(clip_labels)):
                    if (clip_labels[i] == positive_class) != (clip_labels[i - 1] == positive_class):
                        transition_indices.append(i)

                near_transition = np.zeros(len(clip_labels), dtype=bool)
                for t in transition_indices:
                    t_start = max(0, t - transition_margin)
                    t_end = min(len(clip_labels), t + transition_margin + 1)
                    near_transition[t_start:t_end] = True

                eligible_neg = (~gt_pos) & (~near_transition)
                n_eligible = np.sum(eligible_neg)
                fpr = np.sum(eligible_neg & pred_pos_sm) / n_eligible if n_eligible > 0 else float("nan")
                clip_fprs.append(fpr)

                print(f"Clip {clip_idx}: IoU (raw) = {iou_raw:.4f}, IoU (2-of-3) = {iou_sm:.4f}, "
                      f"Hit Rate = {hit_rate:.4f}, FPR = {fpr:.4f}")

            valid_raw = [x for x in clip_ious_raw if not np.isnan(x)]
            valid_sm = [x for x in clip_ious_smoothed if not np.isnan(x)]
            valid_hit = [x for x in clip_hit_rates if not np.isnan(x)]
            valid_fpr = [x for x in clip_fprs if not np.isnan(x)]
            mean_iou_raw = np.mean(valid_raw) if valid_raw else float("nan")
            mean_iou_sm = np.mean(valid_sm) if valid_sm else float("nan")
            mean_hit_rate = np.mean(valid_hit) if valid_hit else float("nan")
            mean_fpr = np.mean(valid_fpr) if valid_fpr else float("nan")

            print(f"\nMean IoU (raw): {mean_iou_raw:.4f}")
            print(f"Mean IoU (2-of-3 rule): {mean_iou_sm:.4f}")
            print(f"Mean Hit Rate: {mean_hit_rate:.4f}")
            print(f"Mean FPR: {mean_fpr:.4f}")

            if not np.isnan(mean_iou_raw):
                self.log("test_iou_raw", mean_iou_raw)
            if not np.isnan(mean_iou_sm):
                self.log("test_iou_realtime", mean_iou_sm)
            if not np.isnan(mean_hit_rate):
                self.log("test_hit_rate", mean_hit_rate)
            if not np.isnan(mean_fpr):
                self.log("test_fpr", mean_fpr)

            # Log per-clip table
            table_data = []
            for clip_idx, (iou_r, iou_s, hr, fp) in enumerate(
                zip(clip_ious_raw, clip_ious_smoothed, clip_hit_rates, clip_fprs)
            ):
                table_data.append([
                    f"Clip {clip_idx}", self.test_clip_sizes[clip_idx],
                    iou_r, iou_s, hr, fp
                ])
            table_data.append([
                "Mean", sum(self.test_clip_sizes),
                mean_iou_raw, mean_iou_sm, mean_hit_rate, mean_fpr
            ])
            iou_table = wandb.Table(
                columns=["Clip", "Frames", "IoU (raw)", "IoU (2-of-3 rule)", "Hit Rate", "FPR"],
                data=table_data
            )
            self.logger.experiment.log({"Realtime Metrics": iou_table})

        self.test_preds = []
        self.test_labels = []

        test_dataloader = self.trainer.datamodule.test_dataloader()
        self.log_test_video(test_dataloader, fps=30)

    def log_test_video(self, dataloader, fps=30, output_size=(640, 480)):
        self.eval()
        all_raw_frames = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _bd in dataloader:
                images = images.to(self.device)
                logits = self(images)
                preds = torch.argmax(logits, dim=1)

                for i in range(len(images)):
                    last_frame = images[i][-1]  # last frame of triplet
                    frame = last_frame.cpu().permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    frame = frame * std + mean
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR)

                    all_raw_frames.append(frame)
                    all_preds.append(preds[i].item())
                    all_labels.append(labels[i].item())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        clip_sizes = self.test_clip_sizes if hasattr(self, "test_clip_sizes") and self.test_clip_sizes is not None else [len(all_preds)]
        positive_class = 1

        offset = 0
        for clip_idx, clip_size in enumerate(clip_sizes):
            clip_frames = all_raw_frames[offset:offset + clip_size]
            clip_preds = all_preds[offset:offset + clip_size]
            clip_labels = all_labels[offset:offset + clip_size]
            offset += clip_size

            realtime_preds = np.zeros_like(clip_preds)
            for i in range(len(clip_preds)):
                start = max(0, i - 2)
                window = clip_preds[start:i + 1]
                realtime_preds[i] = positive_class if np.sum(window == positive_class) >= 2 else 0

            annotated_frames = []
            for i in range(len(clip_frames)):
                frame = clip_frames[i].copy()
                h, w = frame.shape[:2]

                true_label = self.class_names[clip_labels[i]]
                raw_label = self.class_names[clip_preds[i]]
                rt_label = self.class_names[realtime_preds[i]]

                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                raw_color = (0, 255, 0) if raw_label == true_label else (255, 0, 0)
                rt_color = (0, 255, 0) if rt_label == true_label else (255, 0, 0)

                cv2.putText(frame, f"True: {true_label}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Raw: {raw_label}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, raw_color, 2)
                cv2.putText(frame, f"Realtime: {rt_label}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rt_color, 2)

                annotated_frames.append(frame)

            video_array = np.stack(annotated_frames).transpose(0, 3, 1, 2)
            self.logger.experiment.log({
                f"test_video_clip{clip_idx}": wandb.Video(
                    video_array, fps=fps, format="mp4",
                    caption=f"Clip {clip_idx} test predictions ({len(annotated_frames)} frames)"
                )
            })
            print(f"Logged clip {clip_idx} video with {len(annotated_frames)} frames at {fps} fps")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.set_float32_matmul_precision("medium")

    config = OmegaConf.load("config_tsm.yaml")
    wandb_logger = WandbLogger(project=config.project_name, name=config.run_name)

    # Select/early-stop on val_mcc (maximized) — imbalance-robust, tracks decision
    # quality better than the class-weighted val_loss. Keep both callbacks aligned.
    checkpoint_callback = ModelCheckpoint(monitor="val_mcc", mode="max", save_top_k=1)
    early_stopping_callback = EarlyStopping(
        monitor="val_mcc", patience=config.callbacks.early_stopping_patience, mode="max"
    )

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config.max_epochs,
    )

    all_file_paths, all_labels, all_class_names_union, all_boundary_dists = load_data(**config.data_loading)

    train_data, valid_data, test_data, test_clip_sizes = train_test_split(
        all_file_paths, all_labels, all_boundary_dists, **config.data
    )

    class_ids, counts = torch.unique(torch.tensor(train_data[1]), return_counts=True)
    weights = counts.sum() / counts

    print(all_class_names_union, class_ids, counts)
    assert len(counts) == len(all_class_names_union), \
        f"Number of classes ({len(all_class_names_union)}) should equal number of counts ({len(counts)})"

    for weight, name in zip(weights, all_class_names_union):
        print(f"{name}: {weight:.3f}")

    data_module = MultiImageDataModule(train_data, valid_data, test_data)

    boundary_smoothing = (
        OmegaConf.to_container(config.boundary_smoothing, resolve=True)
        if "boundary_smoothing" in config else None
    )

    model = TSMClassifier(
        **config.model,
        max_epochs=config.max_epochs,
        weights=weights,
        class_names=all_class_names_union,
        boundary_smoothing=boundary_smoothing,
    )
    model.test_clip_sizes = test_clip_sizes

    if config.fine_tune_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=data_module, min_lr=1e-6, max_lr=1e-2, num_training=200)
        fig = lr_finder.plot(suggest=True)
        plt.savefig("lr_finder_tsm.png")
    else:
        trainer.fit(model, data_module)

        best_ckpt = checkpoint_callback.best_model_path
        print("Best checkpoint path:", best_ckpt)

        trainer.test(model, data_module, ckpt_path=best_ckpt)


if __name__ == "__main__":
    main()
