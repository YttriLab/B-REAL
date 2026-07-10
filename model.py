import numpy as np
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, MatthewsCorrCoef, ConfusionMatrix
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateFinder
from omegaconf import OmegaConf
from torchvision import models
from timm.layers import DropPath
import albumentations as A
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.tuner import Tuner
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import pandas as pd
import wandb
from collections import defaultdict

def load_data(frame_dirs, label_paths, frames_per_group=3, label_union=None, binarize=None):
    """
    Load data from multiple datasets.
    
    Args:
        frame_dirs: List of paths to processed frame directories or single path string
        label_paths: List of paths to label CSV files or single path string
        frames_per_group: Number of frames per group (default: 3)
        label_union: Dict mapping new label names to lists of original labels to combine.
                    Example: {'exploration': ['rearing', 'sniffing', 'grooming']}
                    Original labels are dropped after creating unions.
        binarize: Name of behavior to isolate for binary classification (can be union label).
                 If specified, creates binary labels: [binarize, 'other']
    
    Returns:
        all_frame_paths: Nested list [dataset][sample] of frame group paths
        all_labels: Nested list [dataset][sample] of labels
        class_names: List of class names (union across all datasets)
    """
    # Handle single dataset case for backward compatibility
    if isinstance(frame_dirs, str):
        frame_dirs = [frame_dirs]
        label_paths = [label_paths]
    
    assert len(frame_dirs) == len(label_paths), "Must provide same number of frame directories and label paths"
    
    all_frame_paths = []  # Will be nested: [dataset][sample]
    all_labels = []  # Will be nested: [dataset][sample]
    all_class_names_union = []
    dataset_info_list = []  # Store info per dataset before remapping
    
    for vid_path, data_path in zip(frame_dirs, label_paths):
        # Load labels
        labels = pd.read_csv(data_path)
        
        # Apply label unions if specified
        if label_union is not None:
            for new_label, original_labels in label_union.items():
                # Validate that all original labels exist
                missing_labels = [lbl for lbl in original_labels if lbl not in labels.columns]
                if missing_labels:
                    raise ValueError(f"Labels {missing_labels} not found in dataset columns: {list(labels.columns)}")
                
                # Create new unified column (union = any of the original labels is active)
                labels[new_label] = labels[original_labels].max(axis=1)
                
                # Drop the original columns
                labels = labels.drop(columns=original_labels)
        
        class_names = list(labels.columns[1:])  # Skip 'time' column
        
        # Build union of class names
        for class_name in class_names:
            if class_name not in all_class_names_union:
                all_class_names_union.append(class_name)

        labels = labels.to_numpy()
        labels_frames = labels[:, 0].astype(int)  # Frame indices from 'time' column

        # Create mapping from frame number to label row index
        frame_to_label_idx = {int(frame_num): idx for idx, frame_num in enumerate(labels_frames)}

        # Discover available frames — supports both flat (N.png) and subfolder (N/{0,1,2}.png) layouts
        all_entries = os.listdir(vid_path)

        # Detect layout: if entries are directories containing PNGs, it's subfolder layout
        sample_entry = next((e for e in all_entries if os.path.isdir(os.path.join(vid_path, e))), None)
        is_flat = sample_entry is None  # all entries are files → flat layout

        if is_flat:
            # Flat layout: vid_path/0.png, 1.png, ...
            all_frame_nums = sorted(
                [int(os.path.splitext(f)[0]) for f in all_entries if f.endswith('.png')],
            )
            min_frame_idx = frames_per_group - 1

            frame_group_paths_indiv = []
            labels_int = []

            for frame_num in all_frame_nums:
                if frame_num < min_frame_idx:
                    continue
                if frame_num not in frame_to_label_idx:
                    continue

                # Construct triplet: [frame_num-(T-1), ..., frame_num-1, frame_num]
                triplet_paths = [
                    os.path.join(vid_path, f"{frame_num - (frames_per_group - 1 - j)}.png")
                    for j in range(frames_per_group)
                ]

                # Verify all frames in the triplet exist
                if all(os.path.exists(p) for p in triplet_paths):
                    frame_group_paths_indiv.append(triplet_paths)
                    label_row_idx = frame_to_label_idx[frame_num]
                    label_row = labels[label_row_idx, 1:]
                    labels_int.append(label_row.argmax())
        else:
            # Subfolder layout (legacy): vid_path/N/{0,1,2}.png
            all_folders = sorted(
                [e for e in all_entries if os.path.isdir(os.path.join(vid_path, e))],
                key=lambda x: int(x),
            )
            min_folder_idx = frames_per_group - 1

            frame_group_paths_indiv = []
            labels_int = []

            for folder_name in all_folders:
                frame_num = int(folder_name)
                if frame_num < min_folder_idx:
                    continue
                if frame_num not in frame_to_label_idx:
                    continue

                paths_full = [
                    os.path.join(vid_path, folder_name, f"{idx}.png")
                    for idx in range(frames_per_group)
                ]
                frame_group_paths_indiv.append(paths_full)
                label_row_idx = frame_to_label_idx[frame_num]
                label_row = labels[label_row_idx, 1:]
                labels_int.append(label_row.argmax())

        labels_int = np.array(labels_int)
        
        # Store the dataset info for later remapping
        dataset_info_list.append({
            'frame_paths': frame_group_paths_indiv,
            'labels': labels_int,
            'class_names': class_names
        })
    
    # Remap all labels to the union class space
    for dataset_info in dataset_info_list:
        labels_int = dataset_info['labels']
        class_names = dataset_info['class_names']
        frame_paths = dataset_info['frame_paths']
        
        # Create mapping from old label indices to new union indices
        label_mapping = {old_idx: all_class_names_union.index(class_names[old_idx]) 
                        for old_idx in range(len(class_names))}
        
        # Remap labels
        remapped_labels = np.array([label_mapping[label] for label in labels_int])
        
        # Add as separate dataset (nested structure)
        all_frame_paths.append(frame_paths)
        all_labels.append(remapped_labels.tolist())
    
    # After loading all datasets, handle missing classes across all datasets
    flat_labels = [label for dataset_labels in all_labels for label in dataset_labels]
    flat_labels = np.array(flat_labels)
    label_ids, counts = np.unique(flat_labels, return_counts=True)

    if len(label_ids) != len(all_class_names_union):
        # Keep only classes that actually appear in the data
        kept_class_names = [all_class_names_union[label_id] for label_id in label_ids]
        
        # Print warnings for removed classes
        for idx, class_name in enumerate(all_class_names_union):
            if idx not in label_ids:
                print(f"WARNING: removing {class_name} due to lack of samples")
        
        all_class_names_union = kept_class_names
        
        # Create mapping from old label IDs to new contiguous indices
        label_mapping = {old_id: new_id for new_id, old_id in enumerate(label_ids)}
        
        # Remap labels in each dataset
        for dataset_idx in range(len(all_labels)):
            dataset_labels = np.array(all_labels[dataset_idx])
            remapped_labels = np.array([label_mapping[label] for label in dataset_labels])
            all_labels[dataset_idx] = remapped_labels.tolist()

    # Apply binarization if specified
    if binarize is not None:
        if binarize not in all_class_names_union:
            raise ValueError(f"Binarize behavior '{binarize}' not found in class names: {all_class_names_union}")
        
        beh_idx = all_class_names_union.index(binarize)
        
        # Apply binary conversion to each dataset
        for dataset_idx in range(len(all_labels)):
            dataset_labels = np.array(all_labels[dataset_idx])
            included_beh_idxs = dataset_labels == beh_idx
            dataset_labels[:] = 0  # All labels become 'other' (0)
            dataset_labels[included_beh_idxs] = 1  # Target behavior becomes 1
            all_labels[dataset_idx] = dataset_labels.tolist()
        
        # Update class names: [0='other', 1=binarize]
        all_class_names_union = ['other', binarize]
    
    # Compute per-sample distance to nearest label transition (for boundary smoothing)
    all_boundary_dists = []
    for dataset_labels in all_labels:
        labels_arr = np.array(dataset_labels)
        # Find transition indices (where label changes from one frame to the next)
        transitions = np.where(labels_arr[1:] != labels_arr[:-1])[0] + 1  # index of first frame of new label

        if len(transitions) == 0:
            # No transitions in this dataset — all frames are far from boundaries
            all_boundary_dists.append(np.full(len(labels_arr), float('inf')).tolist())
        else:
            # For each sample, compute distance to nearest transition
            sample_indices = np.arange(len(labels_arr))
            # Broadcasting: |sample_indices[:, None] - transitions[None, :]| → min over transitions
            dists = np.abs(sample_indices[:, None] - transitions[None, :]).min(axis=1)
            all_boundary_dists.append(dists.tolist())

    total_samples = sum(len(dataset_labels) for dataset_labels in all_labels)
    print(f"\nLoaded {len(frame_dirs)} dataset(s) with {total_samples} total samples")
    print(f"Samples per dataset: {[len(dataset_labels) for dataset_labels in all_labels]}")
    print(f"Class names: {all_class_names_union}")

    # Flatten for distribution calculation
    flat_labels = [label for dataset_labels in all_labels for label in dataset_labels]
    print(f"Class distribution: {dict(zip(*np.unique(flat_labels, return_counts=True)))}")

    flat_dists = [d for dataset_dists in all_boundary_dists for d in dataset_dists]
    flat_dists_finite = [d for d in flat_dists if d != float('inf')]
    if flat_dists_finite:
        print(f"Boundary distances: mean={np.mean(flat_dists_finite):.1f}, median={np.median(flat_dists_finite):.1f}, "
              f"samples within 5 frames of boundary: {sum(1 for d in flat_dists if d <= 5)}/{len(flat_dists)}")

    return all_frame_paths, all_labels, all_class_names_union, all_boundary_dists

# Dataset class
class MultiImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.image_paths, self.labels, self.boundary_dists = data
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img1_path, img2_path, img3_path = self.image_paths[idx]
        label = self.labels[idx]
        boundary_dist = self.boundary_dists[idx]

        # Load images with OpenCV and convert BGR to RGB
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(cv2.imread(img3_path), cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=img1,
                image1=img2,
                image2=img3
            )
            img1 = transformed['image']
            img2 = transformed['image1']
            img3 = transformed['image2']

        triplet = torch.stack([img1, img2, img3])

        return triplet, label, boundary_dist

# DataModule
class MultiImageDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        test_data=None,
        batch_size=64,
        num_workers=4,
        image_size=224,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Training transforms with augmentation
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=180, p=.5, border_mode=cv2.BORDER_REPLICATE),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.2),
            A.GaussNoise(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            A.pytorch.ToTensorV2(),
        ], additional_targets={'image1': 'image', 'image2': 'image'})
        
        # Validation/test transforms without augmentation
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
            A.pytorch.ToTensorV2(),
        ], additional_targets={'image1': 'image', 'image2': 'image'})
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiImageDataset(self.train_data, self.train_transform)
            self.val_dataset = MultiImageDataset(self.val_data, self.val_transform)
        
        if stage == 'test' or stage is None:
            if self.test_data is not None:
                self.test_dataset = MultiImageDataset(self.test_data, self.val_transform)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def test_dataloader(self):
        if self.test_data is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

def _find_bouts(labels_arr: np.ndarray, max_chunk_len: int = 300) -> list:
    """Maximal runs of one label in a dataset's (already binarized/remapped)
    label sequence, subdivided so no single chunk exceeds max_chunk_len.
    Index space is the per-dataset labeled-sample sequence load_data
    produces -- not necessarily contiguous native frame numbers, since only
    labeled (in-bout) frames are kept.

    The subdivision matters for the negative/background class: an actual
    annotated behavior bout is typically short (tens to low hundreds of
    frames), but "other" is frequently one huge contiguous run spanning
    most of a video (everything between/around annotated bouts) -- tens of
    thousands of frames in a single run. Treating that whole run as one
    indivisible unit means train_test_split's greedy bout-to-split
    assignment can only place it entirely in one split, which can single-
    handedly blow a split's target percentage by tens of points (there's
    nothing smaller to fill the remainder with). Capping chunk length gives
    the assignment fine-grained pieces to balance with -- max_chunk_len=300
    (~5s at 60fps) bounds the resulting quantization error to a small
    fraction of any real video, while staying far above frames_per_group so
    it never subdivides an actual short behavior bout (those pass through
    as a single chunk, still never split apart).

    Returns a list of (start, end_exclusive, label).
    """
    bouts = []
    start = 0
    n = len(labels_arr)
    for i in range(1, n + 1):
        if i == n or labels_arr[i] != labels_arr[start]:
            run_start, run_end, lbl = start, i, int(labels_arr[start])
            pos = run_start
            while pos < run_end:
                chunk_end = min(pos + max_chunk_len, run_end)
                bouts.append((pos, chunk_end, lbl))
                pos = chunk_end
            start = i
    return bouts


def train_test_split(file_paths, labels, boundary_dists, train_percent=0.7, val_percent=0.15, test_percent=0.15,
                     shuffle=False, random_seed=42, return_dataset_ids=False):
    """
    Split data from multiple datasets into train/val/test sets.
    Splits each dataset separately, then combines across datasets.

    shuffle=False (chronological, the old default): each video is cut in
    native-frame order -- first train_percent, next val_percent, last
    test_percent. Simple, but behavior bouts cluster in time within a
    session (e.g. grooming can taper off entirely by the end of a
    recording), so val/test can land on wildly different class balances
    than train purely from where in the timeline the cut fell, not from any
    real train/test difference in difficulty.

    shuffle=True: splitting is bout-level and class-stratified instead.
    Each video's labeled-frame sequence is grouped into bouts (maximal runs
    of one label, see _find_bouts); within each (dataset, label) group,
    bouts are shuffled and then greedily assigned to train/val/test by
    cumulative bout length until each split's target percentage of that
    group is reached. This never splits a bout across two sets (avoiding
    near-duplicate frames landing on both sides of a split) while making
    every split's class balance track the source video's real balance,
    rather than depending on session timing. Within each split, samples are
    kept in original chronological order (not row-shuffled) so consecutive
    entries stay temporally meaningful wherever they came from the same bout.

    Trade-off: a bout's first frames_per_group-1 samples draw lead-in frames
    from just before the bout starts (extracted for clip context). Under
    the chronological split that lead-in almost always fell in the same
    split (except right at the one train/val or val/test cut per video).
    Bout-level shuffling creates many more cut points, so lead-in context
    can occasionally cross a split boundary -- a small, bounded amount of
    context overlap traded for far more representative splits.

    Args:
        file_paths: Nested list [dataset][sample] of file paths
        labels: Nested list [dataset][sample] of labels
        boundary_dists: Nested list [dataset][sample] of boundary distances
        train_percent: Percentage of data for training (0-1)
        val_percent: Percentage of data for validation (0-1)
        test_percent: Percentage of data for testing (0-1)
        shuffle: False = chronological split; True = bout-level stratified split
        random_seed: Random seed for reproducibility (only used when shuffle=True)
        return_dataset_ids: If True, also return a per-sample array (parallel
            to y_train) recording which source dataset each training sample
            came from -- needed by callers that stratify/report on top of the
            training pool (e.g. benchmark_size_learning_curve.py), since
            bout-level splitting means a dataset's train rows are no longer
            simply "the first N samples of that dataset" and can't be
            recovered by replaying the split arithmetic after the fact.

    Returns:
        (x_train, y_train, bd_train), (x_valid, y_valid, bd_valid), (x_test, y_test, bd_test), test_clip_sizes
        (..., train_dataset_ids) if return_dataset_ids=True
    """
    assert abs(train_percent + val_percent + test_percent - 1.0) < 1e-6, \
        "train_percent + val_percent + test_percent must equal 1.0"

    x_train, x_valid, x_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    bd_train, bd_valid, bd_test = [], [], []
    test_clip_sizes = []
    train_dataset_ids = []

    # Process each dataset separately
    for dataset_idx, (dataset_paths, dataset_labels, dataset_bd) in enumerate(zip(file_paths, labels, boundary_dists)):
        total = len(dataset_labels)
        labels_arr = np.array(dataset_labels)
        dataset_paths_arr = np.array(dataset_paths, dtype=object)
        dataset_bd_arr = np.array(dataset_bd)

        if shuffle:
            rng = np.random.RandomState(None if random_seed is None else random_seed + dataset_idx)
            bouts = _find_bouts(labels_arr)

            bouts_by_label = defaultdict(list)
            for bout_idx, (_s, _e, lbl) in enumerate(bouts):
                bouts_by_label[lbl].append(bout_idx)

            split_of_bout = {}
            for lbl, bout_idxs in bouts_by_label.items():
                bout_idxs = list(bout_idxs)
                rng.shuffle(bout_idxs)
                lengths = [bouts[i][1] - bouts[i][0] for i in bout_idxs]
                label_total = sum(lengths)
                train_end = round(label_total * train_percent)
                val_end = round(label_total * (train_percent + val_percent))

                cum = 0
                n_val_bouts = n_test_bouts = 0
                for bout_idx, length in zip(bout_idxs, lengths):
                    if cum < train_end:
                        split_of_bout[bout_idx] = "train"
                    elif cum < val_end:
                        split_of_bout[bout_idx] = "val"
                        n_val_bouts += 1
                    else:
                        split_of_bout[bout_idx] = "test"
                        n_test_bouts += 1
                    cum += length

                if len(bout_idxs) > 1 and (n_val_bouts == 0 or n_test_bouts == 0):
                    print(f"  WARNING: dataset {dataset_idx} label {lbl} has only "
                          f"{len(bout_idxs)} bout(s) -- val/test may end up with none "
                          f"of this class from this video (a bout can't be split).")

            train_pos, val_pos, test_pos = [], [], []
            for bout_idx, (s, e, _lbl) in enumerate(bouts):
                target = {"train": train_pos, "val": val_pos, "test": test_pos}[split_of_bout[bout_idx]]
                target.extend(range(s, e))
            train_indices = np.array(sorted(train_pos), dtype=int)
            val_indices = np.array(sorted(val_pos), dtype=int)
            test_indices = np.array(sorted(test_pos), dtype=int)
        else:
            indices = np.arange(total)
            train_end = int(total * train_percent)
            val_end = int(total * (train_percent + val_percent))
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

        # Split this dataset
        x_train.extend(dataset_paths_arr[train_indices].tolist())
        y_train.extend(labels_arr[train_indices].tolist())
        bd_train.extend(dataset_bd_arr[train_indices].tolist())
        train_dataset_ids.extend([dataset_idx] * len(train_indices))

        x_valid.extend(dataset_paths_arr[val_indices].tolist())
        y_valid.extend(labels_arr[val_indices].tolist())
        bd_valid.extend(dataset_bd_arr[val_indices].tolist())

        x_test.extend(dataset_paths_arr[test_indices].tolist())
        y_test.extend(labels_arr[test_indices].tolist())
        bd_test.extend(dataset_bd_arr[test_indices].tolist())

        test_clip_sizes.append(len(test_indices))
        print(f"Dataset {dataset_idx}: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    print(f"\nTotal across all datasets: {len(x_train)} train, {len(x_valid)} val, {len(x_test)} test")

    result = (x_train, y_train, bd_train), (x_valid, y_valid, bd_valid), (x_test, y_test, bd_test), test_clip_sizes
    if return_dataset_ids:
        return result + (np.array(train_dataset_ids, dtype=int),)
    return result

## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        
        return x

class MultiImageTransformerClassifier(L.LightningModule):
    def __init__(self, num_images=3, d_model=512, num_heads=8, num_layers=1,
                 attn_drop_rate=0.1, dropout=0.1, drop_path=0.1, lr=1e-4, no_transformer=False,
                 use_dino=True, weight_decay=1e-3, max_epochs=100, weights=None, class_names=None,
                 boundary_smoothing=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = len(class_names)

        if self.num_classes > 2:
            task = 'multiclass'
        else:
            task = 'binary'

        if use_dino:
            self.feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            feature_size = self.feature_extractor.embed_dim
        else:
            self.feature_extractor = models.efficientnet_b0(weights='DEFAULT')
            feature_size = self.feature_extractor.classifier[1].in_features
            self.feature_extractor.classifier = nn.Identity()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        if not use_dino:
            self.feature_extractor.features[-1] = Conv2dNormActivation(
                in_channels=320,
                out_channels=feature_size,
                kernel_size=1,
                norm_layer=nn.BatchNorm2d,
                activation_layer=nn.ReLU
            )
            for param in self.feature_extractor.features[-1].parameters():
                if not param.requires_grad:
                    raise ValueError("Should be trainable")

        # self.dimension_changer = nn.Linear(feature_size, d_model)

        # self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_images, d_model))
        if no_transformer:
            self.encoder = nn.Sequential(
                nn.Linear(d_model*3, d_model*2),
                nn.BatchNorm1d(d_model*2),
                nn.Dropout(dropout),
                nn.Linear(d_model*2, d_model),
                nn.BatchNorm1d(d_model),
                nn.Dropout(dropout)
            )
        else:
            self.encoder = Transformer(
                embed_dim=d_model,
                depth=num_layers,
                num_heads=num_heads,
                qkv_bias=False,
                drop_rate=dropout,
                drop_path_rate=drop_path,
                attn_drop_rate=attn_drop_rate
            )

        self.decoder = nn.Sequential(
                nn.Linear(d_model, self.num_classes)
                )

        self.criterion = nn.CrossEntropyLoss(weight=weights)

        # Boundary-aware label smoothing config
        bs = boundary_smoothing or {}
        self.boundary_smoothing_enabled = bs.get('enabled', False)
        self.boundary_smoothing_epsilon = bs.get('epsilon', 0.1)
        self.boundary_smoothing_window = bs.get('window', 5)
        # Store class weights for manual soft-label loss computation
        self.register_buffer('class_weights', weights if weights is not None else torch.ones(self.num_classes))

        self.train_acc = Accuracy(task=task, num_classes=self.num_classes, average='macro')
        self.val_acc = Accuracy(task=task, num_classes=self.num_classes, average='macro')
        self.test_acc = Accuracy(task=task, num_classes=self.num_classes, average='macro')
        
        self.train_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.val_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)
        self.test_mcc = MatthewsCorrCoef(task=task, num_classes=self.num_classes)

        self.test_confusion = ConfusionMatrix(task=task, num_classes=self.num_classes)
        self.class_names = class_names or [f"Class_{i}" for i in range(self.num_classes)]

        self.test_preds = []
        self.test_labels = []

    def forward(self, x):
        # x shape: (batch_size, num_images, channels, height, width)
        batch_size, num_images, c, h, w = x.shape
        
        # Reshape to process all images at once
        x = x.view(batch_size * num_images, c, h, w)
        
        # Extract features
        feats = self.feature_extractor(x)
        # feats = self.dimension_changer(feats)
        
        # Reshape back to (batch_size, num_images, d_model)
        feats = feats.view(batch_size, num_images, -1)
        
        # Add CLS token
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # feats = torch.cat([cls_tokens, feats], dim=1)
        
        # Transformer
        if self.hparams.no_transformer:
            feats = feats.view(batch_size, -1)
            feats = self.encoder(feats)
        else:
            feats = self.encoder(feats, self.pos_embedding)
            feats = feats[:, -1]

        logits = self.decoder(feats)
        
        return logits
    
    def _boundary_smoothed_loss(self, logits, labels, boundary_dists):
        """Compute cross-entropy loss with boundary-aware label smoothing."""
        eps = self.boundary_smoothing_epsilon
        window = self.boundary_smoothing_window

        # Per-sample epsilon: linear decay from boundary
        # epsilon_i = eps * max(0, 1 - dist_i / window)
        per_epsilon = eps * (1.0 - boundary_dists.float() / window).clamp(min=0.0)

        # Build soft targets: one_hot * (1 - eps_i) + eps_i / K
        one_hot = F.one_hot(labels, self.num_classes).float()
        soft_targets = one_hot * (1.0 - per_epsilon.unsqueeze(1)) + per_epsilon.unsqueeze(1) / self.num_classes

        # Weighted soft cross-entropy
        log_probs = F.log_softmax(logits, dim=1)
        per_sample_loss = -(soft_targets * log_probs).sum(dim=1)

        # Apply class weights (same weighting as nn.CrossEntropyLoss)
        sample_weights = self.class_weights[labels]
        loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()

        return loss, per_epsilon

    def training_step(self, batch, batch_idx):
        images, labels, boundary_dists = batch
        logits = self(images)

        if self.boundary_smoothing_enabled:
            loss, per_epsilon = self._boundary_smoothed_loss(logits, labels, boundary_dists)
            self.log('train_mean_epsilon', per_epsilon.mean(), on_step=False, on_epoch=True)
            self.log('train_smoothed_frac', (per_epsilon > 0).float().mean(), on_step=False, on_epoch=True)
        else:
            loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_balanced_acc', self.train_acc(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mcc', self.train_mcc(preds, labels), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, boundary_dists = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_balanced_acc', self.val_acc(preds, labels), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mcc', self.val_mcc(preds, labels), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels, boundary_dists = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        self.log('test_loss', loss)
        self.log('test_balanced_acc', self.test_acc(preds, labels), on_step=False, on_epoch=True)
        self.log('test_mcc', self.test_mcc(preds, labels), on_step=False, on_epoch=True)

        self.test_confusion.update(preds, labels)
        self.test_preds.append(preds.cpu())
        self.test_labels.append(labels.cpu())

        return loss

    def on_test_epoch_end(self):
        cm = self.test_confusion.compute().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_normalized, vmin=0, vmax=1)

        plt.colorbar(im, ax=ax, label='Fraction of True')

        ax.set_xticks(range(self.num_classes), labels=self.class_names, rotation=45, ha='right')
        ax.set_yticks(range(self.num_classes), labels=self.class_names)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Test Confusion Matrix')

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
        
        self.logger.experiment.log({"Confusion Matrix - Test Set": wandb.Image(fig)})
        plt.close(fig)
        
        self.test_confusion.reset()

        # Realtime metrics (2-of-3 majority rule, per clip)
        if hasattr(self, 'test_clip_sizes') and self.test_clip_sizes is not None:
            all_preds = torch.cat(self.test_preds).numpy()
            all_labels = torch.cat(self.test_labels).numpy()
            positive_class = 1
            transition_margin = 3  # frames to exclude on each side of onset/offset for FPR

            clip_ious_raw = []
            clip_ious_smoothed = []
            clip_hit_rates = []
            clip_fprs = []

            offset = 0
            for clip_idx, clip_size in enumerate(self.test_clip_sizes):
                clip_preds = all_preds[offset:offset + clip_size]
                clip_labels = all_labels[offset:offset + clip_size]
                offset += clip_size

                gt_pos = clip_labels == positive_class

                # Raw IoU (per-frame argmax, no temporal smoothing)
                pred_pos_raw = clip_preds == positive_class
                intersection_raw = np.sum(gt_pos & pred_pos_raw)
                union_raw = np.sum(gt_pos | pred_pos_raw)
                iou_raw = intersection_raw / union_raw if union_raw > 0 else float('nan')
                clip_ious_raw.append(iou_raw)

                # Apply 2-of-3 majority rule (positive if >= 2 of last 3 frames are positive)
                smoothed_preds = np.zeros_like(clip_preds)
                for i in range(len(clip_preds)):
                    start = max(0, i - 2)
                    window = clip_preds[start:i + 1]
                    smoothed_preds[i] = positive_class if np.sum(window == positive_class) >= 2 else 0

                pred_pos_smoothed = smoothed_preds == positive_class
                intersection_sm = np.sum(gt_pos & pred_pos_smoothed)
                union_sm = np.sum(gt_pos | pred_pos_smoothed)
                iou_smoothed = intersection_sm / union_sm if union_sm > 0 else float('nan')
                clip_ious_smoothed.append(iou_smoothed)

                # Hit rate: bout-level recall — fraction of GT positive bouts
                # where at least one frame is classified positive by the realtime rule
                bout_starts = []
                bout_ends = []
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
                    hit_rate = float('nan')
                clip_hit_rates.append(hit_rate)

                # False positive rate: fraction of clearly-negative frames
                # (GT negative AND more than `transition_margin` frames from any
                # positive-class onset/offset) where the realtime rule predicts positive
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
                if n_eligible > 0:
                    fpr = np.sum(eligible_neg & pred_pos_smoothed) / n_eligible
                else:
                    fpr = float('nan')
                clip_fprs.append(fpr)

                print(f"Clip {clip_idx}: IoU (raw) = {iou_raw:.4f}, IoU (2-of-3) = {iou_smoothed:.4f}, "
                      f"Hit Rate = {hit_rate:.4f}, FPR = {fpr:.4f}")

            valid_raw = [x for x in clip_ious_raw if not np.isnan(x)]
            valid_smoothed = [x for x in clip_ious_smoothed if not np.isnan(x)]
            valid_hit = [x for x in clip_hit_rates if not np.isnan(x)]
            valid_fpr = [x for x in clip_fprs if not np.isnan(x)]
            mean_iou_raw = np.mean(valid_raw) if valid_raw else float('nan')
            mean_iou_smoothed = np.mean(valid_smoothed) if valid_smoothed else float('nan')
            mean_hit_rate = np.mean(valid_hit) if valid_hit else float('nan')
            mean_fpr = np.mean(valid_fpr) if valid_fpr else float('nan')

            print(f"\nMean IoU (raw): {mean_iou_raw:.4f}")
            print(f"Mean IoU (2-of-3 rule): {mean_iou_smoothed:.4f}")
            print(f"Mean Hit Rate: {mean_hit_rate:.4f}")
            print(f"Mean FPR: {mean_fpr:.4f}")

            if not np.isnan(mean_iou_raw):
                self.log('test_iou_raw', mean_iou_raw)
            if not np.isnan(mean_iou_smoothed):
                self.log('test_iou_realtime', mean_iou_smoothed)
            if not np.isnan(mean_hit_rate):
                self.log('test_hit_rate', mean_hit_rate)
            if not np.isnan(mean_fpr):
                self.log('test_fpr', mean_fpr)

            # Log per-clip table to wandb
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
                mean_iou_raw, mean_iou_smoothed, mean_hit_rate, mean_fpr
            ])

            iou_table = wandb.Table(
                columns=["Clip", "Frames", "IoU (raw)", "IoU (2-of-3 rule)",
                         "Hit Rate", "FPR"],
                data=table_data
            )
            self.logger.experiment.log({"Realtime Metrics": iou_table})

        self.test_preds = []
        self.test_labels = []

        test_dataloader = self.trainer.datamodule.test_dataloader()
        self.log_test_video(test_dataloader, fps=30)
    
    def log_test_video(self, dataloader, fps=30, output_size=(640, 480)):
        """
        Create per-clip videos from test set predictions.
        Each frame is annotated with ground truth, raw model prediction,
        and the realtime prediction (2-of-3 majority rule).

        Args:
            dataloader: Test dataloader (with shuffle=False)
            fps: Frames per second for the output video
            output_size: (width, height) for output frames
        """

        self.eval()
        all_raw_frames = []   # decoded image arrays
        all_preds = []        # raw per-frame predictions (int)
        all_labels = []       # ground truth labels (int)

        # Collect all frames and raw predictions in order
        with torch.no_grad():
            for batch_idx, (images, labels, _boundary_dists) in enumerate(dataloader):
                images = images.to(self.device)
                logits = self(images)
                preds = torch.argmax(logits, dim=1)

                for i in range(len(images)):
                    # Get the last frame (index 2) from the triplet
                    last_frame = images[i][2]  # Shape: (C, H, W)

                    # Denormalize
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

        # Split by clip and generate one video per clip
        clip_sizes = self.test_clip_sizes if hasattr(self, 'test_clip_sizes') and self.test_clip_sizes is not None else [len(all_preds)]
        positive_class = 1

        offset = 0
        for clip_idx, clip_size in enumerate(clip_sizes):
            clip_frames = all_raw_frames[offset:offset + clip_size]
            clip_preds = all_preds[offset:offset + clip_size]
            clip_labels = all_labels[offset:offset + clip_size]
            offset += clip_size

            # Compute realtime predictions (2-of-3 majority rule) for this clip
            realtime_preds = np.zeros_like(clip_preds)
            for i in range(len(clip_preds)):
                start = max(0, i - 2)
                window = clip_preds[start:i + 1]
                realtime_preds[i] = positive_class if np.sum(window == positive_class) >= 2 else 0

            # Annotate each frame
            annotated_frames = []
            for i in range(len(clip_frames)):
                frame = clip_frames[i].copy()
                h, w = frame.shape[:2]

                true_label = self.class_names[clip_labels[i]]
                raw_label = self.class_names[clip_preds[i]]
                rt_label = self.class_names[realtime_preds[i]]

                # Semi-transparent background for 3 lines of text
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

                # Color code: green if matches ground truth, red if wrong
                raw_color = (0, 255, 0) if raw_label == true_label else (255, 0, 0)
                rt_color = (0, 255, 0) if rt_label == true_label else (255, 0, 0)

                cv2.putText(frame, f"True: {true_label}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Raw: {raw_label}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, raw_color, 2)
                cv2.putText(frame, f"Realtime: {rt_label}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, rt_color, 2)

                annotated_frames.append(frame)

            # Convert to video format: (num_frames, C, H, W)
            video_array = np.stack(annotated_frames).transpose(0, 3, 1, 2)

            self.logger.experiment.log({
                f"test_video_clip{clip_idx}": wandb.Video(
                    video_array,
                    fps=fps,
                    format="mp4",
                    caption=f"Clip {clip_idx} test predictions ({len(annotated_frames)} frames)"
                )
            })

            print(f"Logged clip {clip_idx} video with {len(annotated_frames)} frames at {fps} fps")
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        out = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }

        return out

def main():
    torch.set_float32_matmul_precision('medium')
    
    config = OmegaConf.load('config.yaml')
    wandb_logger = WandbLogger(project=config.project_name, name=config.run_name)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=config.callbacks.early_stopping_patience, mode='min')

    trainer = L.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config.max_epochs,
    )

    all_file_paths, all_labels, all_class_names_union, all_boundary_dists = load_data(**config.data_loading)

    train_data, valid_data, test_data, test_clip_sizes = train_test_split(all_file_paths, all_labels, all_boundary_dists, **config.data)

    class_ids, counts = torch.unique(torch.tensor(train_data[1]), return_counts=True)
    weights = counts.sum() / counts

    print(all_class_names_union, class_ids, counts)

    assert len(counts) == len(all_class_names_union), f"Number of classes ({len(all_class_names_union)}) should equal number of counts ({len(counts)})"

    for weight, name in zip(weights, all_class_names_union):
        print(f"{name}: {weight:.3f}")

    data_module = MultiImageDataModule(train_data, valid_data, test_data)

    boundary_smoothing = OmegaConf.to_container(config.boundary_smoothing, resolve=True) if 'boundary_smoothing' in config else None

    model = MultiImageTransformerClassifier(**config.model, max_epochs=config.max_epochs, weights=weights,
                                            class_names=all_class_names_union, boundary_smoothing=boundary_smoothing)
    model.test_clip_sizes = test_clip_sizes

    if config.fine_tune_lr:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(model, datamodule=data_module, min_lr=1e-6, max_lr=1e-2, num_training=200)
        fig = lr_finder.plot(suggest=True)
        plt.savefig('lr_finder.png')
    else:
        trainer.fit(model, data_module)

        best_ckpt = checkpoint_callback.best_model_path
        print("Best checkpoint path:", best_ckpt)

        trainer.test(model, data_module, ckpt_path=best_ckpt)

if __name__ == "__main__":
    main()

