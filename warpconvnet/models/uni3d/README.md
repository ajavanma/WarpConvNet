# Uni3D

WarpConvNet provides native implementations of the released
[BAAI Uni3D](https://github.com/baaivision/Uni3D) point encoders. The models load the published
Hugging Face checkpoints directly and do not require the Uni3D source tree, `timm`, or
`pointnet2_ops` at runtime.

Uni3D and PPAT are separate model families. They use different point tokenizers, transformer
architectures, output spaces, and text teachers. A Uni3D shape embedding must be compared with
text features from its paired EVA-CLIP-E teacher, not with PPAT text features.

WarpConvNet ships the Uni3D point encoders and checkpoint loaders. It does not bundle the paired
EVA-CLIP-E text teacher or precomputed label features. Use the
[upstream Uni3D zero-shot evaluation workflow](https://github.com/baaivision/Uni3D#evaluation-of-zero-shot-3d-classification)
to create text features for open-vocabulary label ranking.

The adapted Uni3D implementation retains its upstream MIT license; see [LICENSE](LICENSE).
Checkpoint files are distributed separately by BAAI, whose Hugging Face repository does not state
a checkpoint license.

## Supported models

All scales use a Y-up coordinate frame, 512 farthest-point-sampled centers, 64 neighbors per
center, and a 1,024-value output. The model configuration selects the matching EVA transformer
layout and checkpoint.

| Aliases      | Backbone    | Up axis | Point parameters | Checkpoint size |
| ------------ | ----------- | ------- | ---------------: | --------------: |
| `ti`, `tiny` | EVA02-Ti/14 | Y       |        6,384,449 |         12.8 MB |
| `s`, `small` | EVA02-S/14  | Y       |       22,823,681 |         45.7 MB |
| `b`, `base`  | EVA02-B/14  | Y       |       88,960,617 |        178.0 MB |
| `l`, `large` | EVA02-L/14  | Y       |      307,350,313 |        614.9 MB |
| `g`, `giant` | EVA-g/14    | Y       |    1,017,358,057 |      2,034.9 MB |

Each scale includes the standard ensembled checkpoint and a `-no-lvis` checkpoint. Giant also
includes the released LVIS, ModelNet40, and ScanObjectNN task checkpoints. `UNI3D_VARIANTS` lists
all canonical names.

## Usage

Install the shared demo dependencies to enable checkpoint downloads:

```bash
pip install -e ".[demo]"
```

Load a released model by scale or canonical variant name:

```python
import torch
import torch.nn.functional as F

from warpconvnet.models import load_uni3d

model = load_uni3d("small", device="cuda")

# XYZ must be Y-up for every Uni3D scale.
xyz = torch.randn(1, 10_000, 3, device="cuda")
xyz = xyz - xyz.mean(dim=1, keepdim=True)
xyz = xyz / xyz.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
rgb = torch.rand_like(xyz)
points = torch.cat((xyz, rgb), dim=-1)

with torch.inference_mode():
    shape_embedding = F.normalize(model(points), dim=-1)
```

Inputs have shape `(batch, points, 6)`, with Y-up, centered, unit-ball-scaled XYZ followed by RGB
in `[0, 1]`. Rotate other coordinate frames to Y-up before centering and scaling; neither the model
nor normalization rotates coordinates. The published evaluation format uses 10,000 points and
preserves their stored row order. Row order affects farthest-point sampling because sampling begins
from the first point. `UNI3D_UP_AXIS` and every `Uni3DVariant.up_axis` expose this contract as
`"y"`.

`load_uni3d` strictly loads the selected checkpoint and returns an evaluation-mode model.
`checkpoint_path` selects local weights, `verify_download=True` verifies a downloaded checkpoint,
and `dtype=None` preserves the checkpoint dtype. `build_uni3d` creates an uninitialized model.
Models are constructed on the meta device while loading, avoiding a second randomly initialized
copy of large checkpoints.

## Neighborhood backends

The neighbor selector is configured with `neighbor_search`:

| Backend    | Device   | Behavior                                                        |
| ---------- | -------- | --------------------------------------------------------------- |
| `knn`      | CPU/CUDA | Exact released selector and default                             |
| `cell-knn` | CUDA     | Fused cell-list selector with query-selective exact fallback    |
| Callable   | Any      | Custom selector returning `(batch, centers, neighbors)` indices |

```python
exact = load_uni3d("small", device="cuda", neighbor_search="knn")
fused = load_uni3d("small", device="cuda", neighbor_search="cell-knn")
```

The fused selector uses cubic cells, direct FP32 distances, and deterministic `(distance, point index)` ordering. Its default `cell_size=0.10` and `max_shell=4` are tuned for centered unit-ball
object clouds with 10,000 points. Queries for which the kernel cannot certify the complete nearest
set are recomputed by the exact selector. `cell-knn` supports at most 64 neighbors.

The exact selector uses an algebraically expanded distance expression, while the fused kernel uses
direct subtraction. FP32 rounding at the 64th-neighbor boundary can therefore select a different,
geometrically equivalent point. Use `knn` when strict released-implementation parity is required.

## Accuracy

The following Uni3D-S results use 8,511 Objaverse-LVIS shapes, all 1,156 labels, FP32 inference,
10,000 points per shape, 512 centers, and 64 neighbors. Only the neighborhood backend differs.

| Backend                     |   Top-1 |   Top-3 |   Top-5 | Macro top-1 | Mean embedding cosine | Top-label agreement |
| --------------------------- | ------: | ------: | ------: | ----------: | --------------------: | ------------------: |
| `knn`                       | 37.763% | 60.381% | 68.970% |     37.338% |           1.000000000 |            100.000% |
| `cell-knn` + exact fallback | 37.763% | 60.381% | 68.970% |     37.338% |           0.999999826 |             99.988% |

Across 4,357,632 patch queries, the fused path directly certified 99.9876% of queries and used
exact fallback for 0.0124%. Its final neighborhoods had 99.9995% recall and a 99.9714% exact-set
rate relative to `knn`. Both backends produced the same aggregate top-1, top-3, top-5, and macro
top-1 accuracy.

## Speed and memory

Measurements use an NVIDIA RTX PRO 6000 Blackwell (`sm_120`), Uni3D-S in FP32, fixed real
10,000-point clouds, 512 queries, 64 neighbors, 10 warmups, and the median of 50 CUDA-event trials.
Peak memory is incremental allocated memory over the pre-call baseline.

### Neighbor selector

| Batch | `knn` time | `cell-knn` time | Speedup |  `knn` peak | `cell-knn` peak |
| ----: | ---------: | --------------: | ------: | ----------: | --------------: |
|     1 |   0.135 ms |        0.578 ms |   0.23x |    40.0 MiB |         1.2 MiB |
|     4 |   0.409 ms |        0.835 ms |   0.49x |   156.4 MiB |         4.8 MiB |
|    16 |   2.775 ms |        1.005 ms |   2.76x |   625.6 MiB |        19.2 MiB |
|    32 |   5.475 ms |        1.148 ms |   4.77x | 1,253.4 MiB |        39.6 MiB |

`cell-knn` reduces selector peak memory by 96.8-97.0%. Its fixed hashing and launch cost makes it
slower at batch 1 and 4; it is faster at batch 16 and 32.

### Full point encoder

| Batch | `knn` time | `cell-knn` time | Latency change |  `knn` peak | `cell-knn` peak |
| ----: | ---------: | --------------: | -------------: | ----------: | --------------: |
|     1 |   3.797 ms |        4.337 ms |         +14.2% |   193.9 MiB |       193.9 MiB |
|     4 |   7.914 ms |        8.406 ms |          +6.2% |   775.4 MiB |       775.4 MiB |
|    16 |  26.937 ms |       25.249 ms |          -6.3% | 3,101.8 MiB |     3,101.8 MiB |
|    32 |  53.550 ms |       49.294 ms |          -7.9% | 6,203.5 MiB |     6,203.5 MiB |

The fused backend improves full-encoder latency at batch 16 and 32. Full-model peak memory is
unchanged because transformer activations are larger than the neighborhood workspace.

## Backend constraints

- `knn` is the portable default and supports CPU and CUDA tensors.
- `cell-knn` requires CUDA and the WarpConvNet native extension.
- Search cost and fallback rate depend on point density and spatial scale. Nonuniform scenes or a
  different normalization can move the performance crossover.
- Exact fallback preserves complete neighborhoods but can remove the speed advantage when many
  queries cannot be certified by the configured cell search.
- The accuracy and performance tables cover Uni3D-S; the other scales use the same grouping API
  but have different transformer costs and memory requirements.
