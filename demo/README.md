# CalMS21 demo project

`calms21_demo/` is a small, ready-to-open B-REAL project: four one-minute clips
of a resident (black) and intruder (white) mouse from the CalMS21 Task 1
training set, with the dataset's expert labels as bouts.

| Video | Source (CalMS21 train) | Labels |
|---|---|---|
| `mouse070_f4200` | mouse070, frames 4200–5999 | attack, investigation, mount, other |
| `mouse018_f1350` | mouse018, frames 1350–3149 | attack, investigation, mount, other |
| `mouse055_f4500` | mouse055, frames 4500–6299 | investigation, mount, other |
| `mouse001_f1350` | mouse001, frames 1350–3149 | none — for trying Active Learning |

The expert labels for the unlabeled clip are in `ground_truth/`, so you can
compare the model's predictions against them (or import them in the Label tab).

Clips are 30 fps, grayscale, scaled to 512 px wide (the originals are
1024×570) and compressed hard so the demo fits in git. Behaviors are keys 1–4:
attack, investigation, mount, other (CalMS21's Task 1 classes; "other" is
everything else).

## Try it

1. `pixi run gui` → **Open project** → `demo/calms21_demo`.
2. **Preprocess** the three labeled videos.
3. **Train** on them with the default settings (about 15 min on a fast GPU).
4. Open `mouse001_f1350` and run **Active Learning** with the new model.

## Expected results

One run of `pixi run python demo/run_calms21_demo.py` (the steps above with
the Train tab's defaults) on an RTX 5090:

| Step | Time |
|---|---|
| Preprocess 3 clips | 14 s |
| Train (early stopping, 28 epochs) | 15 min |
| Active Learning inference, 1 clip | 48 s |
| **Total** | **~16 min** |

Predictions on `mouse001_f1350` against its expert labels (1,798 frames):

| | Balanced accuracy | Macro F1 | Accuracy |
|---|---|---|---|
| All classes | **0.62** | 0.64 | 0.66 |

Per class (recall / F1): investigation 0.65 / 0.66, mount 0.32 / 0.47,
other 0.90 / 0.79. The clip has no attack. Mount is the weak class, since the
training clips have only ~560 mount frames. These are from a single run;
expect some variation between runs. Training takes longer on smaller GPUs and much longer on CPU
(roughly 3 min per epoch on a 32-core machine). With three minutes of training
video this is a pipeline check, not a benchmark: trained on the full CalMS21
training set, the same model reaches about 0.81–0.86 balanced accuracy on the
CalMS21 test set.

What you generate (frames, models, predictions) stays in the folder but is
git-ignored. Opening the project rewrites the video paths in `project.json` for
your machine; that change is harmless.

## Rebuilding

`make_calms21_demo.py` makes the folder from a local copy of CalMS21 Task 1
(videos plus one-hot label CSVs). Edit `CLIPS` in it to choose other windows:

```bash
pixi run python demo/make_calms21_demo.py --calms21 /path/to/Calms21
```

## Data credit and license

The clips and labels are a subset of **CalMS21** (Caltech Mouse Social
Interactions 2021), released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The clips are
trimmed, scaled down and re-encoded, and the per-frame labels are converted
to bouts.

> Sun, J. J., Karigo, T., Chakraborty, D., Mohanty, S. P., Wild, B., Sun, Q.,
> Chen, C., Anderson, D. J., Perona, P., Yue, Y., & Kennedy, A. (2021).
> The Multi-Agent Behavior Dataset: Mouse Dyadic Social Interactions.
> *NeurIPS Datasets and Benchmarks.* Data: https://data.caltech.edu/records/s0vdx-0k302
