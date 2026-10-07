# Bundled localizer weights

`mouse_topdown.pt` is the default for crop method **SAM2 localizer** when no
weights file is chosen (`localizer.DEFAULT_WEIGHTS`). It is a plain PyTorch
state dict for `localizer.Localizer` (ResNet18 trunk + stride-8 heatmap +
presence head, 2.9 M parameters, fp32).

## What it was trained on

One rig: a single mouse, filmed top-down in infrared (monochrome) at
2248×2064, in a clear-walled arena with a tethered headstage. It was distilled
from SAM2 (`sam2.1-hiera-tiny`, one positive click per clip, tail removed by
eroding the mask) using:

- SAM2 body centres for 10 minutes of one session;
- 27 clips from other days, 1000 frames each, seeded by an earlier localizer,
  keeping only frames where SAM2 and that localizer agreed;
- empty-arena frames plus synthetic "animal pasted over" frames, which teach
  the presence head.

On two held-out days, the median centre error was 12–30 px, measured on the
450 px crop box. Classifier accuracy with these crops was within 0.005 MCC of
the SAM2 crops it was distilled from.

## When to train your own

Other rigs, species, lighting, resolutions or camera angles are out of
distribution. Check the Preprocess preview first. If the box misses the animal,
or the heatmap lights up on the wrong thing, train a localizer for your rig in
**Pipeline → Train Localizer**. That step runs SAM2 on clips from your videos
and distils the result into a new network, fine-tuned from these weights by
default. The result is saved under the project's `models/localizer/`.
