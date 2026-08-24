# PPAT: Point Patch Transformer

PPAT turns a colored 3D point cloud into one feature vector. OpenShape trained its released models
to align that vector with CLIP text features. This lets you compare a shape with names such as
"chair", "table", or "car" without training a new classifier.

OpenShape calls the encoder `PointBERT` in its configuration files. It first groups points into
local patches with PointNet++, then combines the patches with a transformer. The L14 and bigG
variants add a linear projection to match their CLIP width; B32 does not need a separate projection
layer. Parameter names match the [OpenShape reference](https://github.com/Colin97/OpenShape_code),
so its published checkpoints load directly.

## Install

Follow the main [WarpConvNet installation guide](../../../README.md#installation) for a compatible
PyTorch and CUDA build. A CUDA GPU is strongly recommended for the large `vitg14-rgb` model. From
a source checkout, initialize CUTLASS and install the two demo dependencies:

```bash
git submodule update --init 3rdparty/cutlass
MAX_JOBS=4 NVCC_THREADS=1 pip install -e ".[demo]" --no-build-isolation
```

The shared `demo` extra contains `huggingface-hub` for model downloads and `trimesh` for mesh I/O.
The script and mesh are source-tree examples rather than installed console commands, so run them
from a checkout. If WarpConvNet is already installed, you can instead run
`pip install huggingface-hub trimesh` in that checkout.

## Run the demo

The default demo uses the bundled z-up chair in `examples/assets/ppat_demo_chair.obj`. It samples
10,000 colored points, normalizes them, and downloads
[`OpenShape/openshape-pointbert-vitg14-rgb`](https://huggingface.co/OpenShape/openshape-pointbert-vitg14-rgb).
For the text side, it downloads OpenShape's small, prompt-averaged ViT-bigG-14 LVIS feature file
from `OpenShape/openshape-demo-support`; it does not download a separate CLIP model.

```bash
python examples/ppat_demo.py
```

Example output for the bundled mesh with 10,000 points and seed 0:

```text
Ranked labels (cosine similarity; higher is a closer text match):
   1. chair     +0.1379
   2. table     +0.0681
   3. sofa      +0.0453
   4. lamp      +0.0161
   5. airplane  +0.0132
   6. guitar    +0.0059
   7. car       -0.0339

Top prediction: chair
```

Cosine similarities are relative scores, not probabilities. Compare labels within one run; do not
read `0.1379` as 13.79% confidence. The demo fixes the mesh-sampling seed and uses the portable
PyTorch farthest-point sampler for stable point selection.

To use your own mesh and candidate labels:

```bash
python examples/ppat_demo.py \
    --mesh path/to/shape.obj \
    --labels chair table airplane guitar \
    --up-axis y
```

`--up-axis` is the mesh's current gravity axis; the script rotates it to the checkpoint's z-up
convention. Labels must be exact OpenShape LVIS identifiers. The CLI also accepts `--num-points`,
`--seed`, and `--device`; use `--help` for details. Downloads are cached by Hugging Face.

## Input and public API

The simplest input is a floating-point tensor shaped `(B, 6, N)`:

```text
channels 0..2: XYZ coordinates
channels 3..5: RGB values in [0, 1]
```

Every item in a tensor batch must have the same `N`. Divide byte RGB values by 255. If a mesh has
no color, OpenShape recommends filling all three RGB channels with `0.4`. PPAT also accepts separate
channels-first XYZ and feature tensors, or a WarpConvNet `Points` object. For those forms, the
feature tensor must still contain the full six-channel XYZ+RGB model input.

Orientation is checkpoint-specific:

| Variant                          | Up axis | Output width |
| -------------------------------- | ------- | -----------: |
| `openshape-pointbert-vitb32-rgb` | Y       |          512 |
| `openshape-pointbert-vitl14-rgb` | Y       |          768 |
| `openshape-pointbert-vitg14-rgb` | Z       |        1,280 |

B32 and L14 are Y-up; bigG is Z-up. These conventions belong to the released checkpoints and do
not change with the neighborhood backend. Rotate into the correct orientation first. Then center
XYZ by subtracting its mean and divide by the greatest point distance from the center.
`normalize_point_cloud` performs those last two steps; it does not rotate the cloud. The required
axis is available as `OPENSHAPE_VARIANTS[variant]["up_axis"]`.

```python
import torch
from warpconvnet.models.ppat import load_openshape_pointbert, normalize_point_cloud

device = "cuda"
model = load_openshape_pointbert(
    "openshape-pointbert-vitg14-rgb", device=device
)

# xyz and rgb are (B, N, 3); xyz is already z-up for this checkpoint.
xyz = normalize_point_cloud(xyz)
points = torch.cat([xyz, rgb], dim=-1).permute(0, 2, 1).contiguous()

with torch.inference_mode():
    shape_embedding = model(points.to(device))  # (B, 1280)
```

With no `checkpoint_path`, `load_openshape_pointbert` downloads `model.pt`, strictly loads it, and
returns an evaluation-mode model. Pass `checkpoint_path="/path/to/model.pt"` to use a local file.
Use `build_openshape_pointbert(variant, in_dim=6)` when you want a newly initialized architecture
instead of published weights.

## Neighborhood backends

The default `neighborhood="cumsum"` uses the same point-selection order as OpenShape's dense ball
query and provides published-implementation parity. Other choices are:

- `dense`: the slower reference implementation; bit-identical to `cumsum`.
- `warp`: exact CUDA cell-list radius search with lower memory, but not lower latency.
- `ball_cuda`: the same spherical rule in a fast, low-memory CUDA kernel.
- `voxel:SIZE` and `voxel_cuda:SIZE`: approximate 3x3x3 voxel blocks. The cell size is required.
- `l1`, `linf`, `random`, and `knn`: optional approximate rules.

`warp` and `ball_cuda` compute distance directly, while the reference-compatible path uses an
expanded floating-point formula. They follow the same geometric rule but are not guaranteed to be
bit-identical at the radius boundary.

Approximate rules can change accuracy. Their useful spatial scale depends on the same input
normalization used for training. The neighborhood rule is not stored in a state dict, so keep the
configuration with the checkpoint and use the same rule for evaluation and deployment. The active
configuration is available through `model.ppat_config_report`.

## Accuracy and performance

The [OpenShape checkpoint table](https://github.com/Colin97/OpenShape_code#checkpoints) reports
46.8% Objaverse-LVIS zero-shot top-1 accuracy for `pointbert-vitg14-rgb`.

The latency measurements below use the supported WarpConvNet path in BF16 on an NVIDIA RTX PRO
6000 Blackwell (`sm_120`).

| Batch 32, 10,000 points |  Latency |
| ----------------------- | -------: |
| Whole model             | 17.06 ms |
| Transformer             |  4.69 ms |

The transformer uses PyTorch scaled dot-product attention with scaled positional bias.

| Ball query, B=96, N=10,000, S=384, K=64 |     Time | Peak memory |
| --------------------------------------- | -------: | ----------: |
| `voxel_cuda:0.1` (approximate)          |  0.53 ms |    0.10 GiB |
| `ball_cuda` (exact sphere)              |  0.79 ms |    0.10 GiB |
| `cumsum` (portable default)             | 20.25 ms |    3.09 GiB |
| `dense` (reference)                     | 25.30 ms |    9.63 GiB |
