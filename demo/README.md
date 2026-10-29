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
3. **Train** on them (a few epochs is enough to see it work).
4. Open `mouse001_f1350` and run **Active Learning** with the new model.

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
