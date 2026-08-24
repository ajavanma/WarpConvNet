# Uni3D

Uni3D is an open-vocabulary point-cloud encoder that maps colored point clouds into the embedding
space of its paired EVA-CLIP-E text teacher. WarpConvNet provides native Tiny, Small, Base, Large,
and Giant implementations that load the released BAAI checkpoints without the Uni3D source tree,
`timm`, or `pointnet2_ops` at runtime.

Uni3D is separate from [PPAT](ppat.md). The models use different tokenizers, transformers, output
spaces, and text features.

WarpConvNet ships the Uni3D point encoders and checkpoint loaders. It does not bundle the paired
EVA-CLIP-E text teacher or precomputed label features; create those features with the
[upstream Uni3D zero-shot evaluation workflow](https://github.com/baaivision/Uni3D#evaluation-of-zero-shot-3d-classification)
before computing text-to-shape similarity.

## Supported models

| Aliases      | Backbone    | Up axis |    Parameters | Checkpoint | Output width |
| ------------ | ----------- | ------- | ------------: | ---------: | -----------: |
| `ti`, `tiny` | EVA02-Ti/14 | Y       |     6,384,449 |    12.8 MB |        1,024 |
| `s`, `small` | EVA02-S/14  | Y       |    22,823,681 |    45.7 MB |        1,024 |
| `b`, `base`  | EVA02-B/14  | Y       |    88,960,617 |   178.0 MB |        1,024 |
| `l`, `large` | EVA02-L/14  | Y       |   307,350,313 |   614.9 MB |        1,024 |
| `g`, `giant` | EVA-g/14    | Y       | 1,017,358,057 | 2,034.9 MB |        1,024 |

Every scale has a standard ensembled checkpoint and a `-no-lvis` checkpoint. Giant also has the
released LVIS, ModelNet40, and ScanObjectNN task checkpoints.

## Usage

```bash
pip install -e ".[demo]"
```

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
    embedding = F.normalize(model(points), dim=-1)
```

Input tensors use `(batch, points, 6)`: Y-up, centered, unit-ball-scaled XYZ followed by RGB in
`[0, 1]`. Tiny, Small, Base, Large, and Giant all use Y-up. Rotate other frames before centering and
scaling; the model does not rotate coordinates. The released format uses 10,000 points, 512
farthest-point-sampled centers, and 64 neighbors per center. The output has width 1,024.

`load_uni3d` downloads and strictly loads released weights. Use `checkpoint_path` for local weights,
`verify_download=True` for checkpoint verification, or `build_uni3d` for an uninitialized model.

## Neighborhood backends

| `neighbor_search` | Device   | Description                                  |
| ----------------- | -------- | -------------------------------------------- |
| `knn`             | CPU/CUDA | Exact released selector and default          |
| `cell-knn`        | CUDA     | Fused cell-list selector with exact fallback |
| Callable          | Any      | Custom index selector                        |

```python
fused = load_uni3d("small", device="cuda", neighbor_search="cell-knn")
```

`cell-knn` uses direct FP32 distances and deterministic ordering. Queries that cannot be certified
within its cell-search budget are recomputed by `knn`. It supports at most 64 neighbors. The default
cell configuration targets centered unit-ball object clouds with 10,000 points; density and scale
affect its work and fallback rate.

## Measured results

On 8,511 Objaverse-LVIS shapes and 1,156 labels, Uni3D-S produced the following results:

| Backend                     |   Top-1 |   Top-3 |   Top-5 | Macro top-1 | Mean embedding cosine |
| --------------------------- | ------: | ------: | ------: | ----------: | --------------------: |
| `knn`                       | 37.763% | 60.381% | 68.970% |     37.338% |           1.000000000 |
| `cell-knn` + exact fallback | 37.763% | 60.381% | 68.970% |     37.338% |           0.999999826 |

The fused backend directly certified 99.9876% of 4,357,632 patch queries and used exact fallback
for 0.0124%. Aggregate top-1, top-3, top-5, and macro top-1 accuracy were unchanged.

Measurements below use Uni3D-S in FP32 on an NVIDIA RTX PRO 6000 Blackwell with 10,000 points,
512 centers, 64 neighbors, and the median of 50 trials.

| Batch | `knn` selector | `cell-knn` selector | Selector speedup | `knn` selector peak | `cell-knn` selector peak |
| ----: | -------------: | ------------------: | ---------------: | ------------------: | -----------------------: |
|     1 |       0.135 ms |            0.578 ms |            0.23x |            40.0 MiB |                  1.2 MiB |
|     4 |       0.409 ms |            0.835 ms |            0.49x |           156.4 MiB |                  4.8 MiB |
|    16 |       2.775 ms |            1.005 ms |            2.76x |           625.6 MiB |                 19.2 MiB |
|    32 |       5.475 ms |            1.148 ms |            4.77x |         1,253.4 MiB |                 39.6 MiB |

| Batch | `knn` encoder | `cell-knn` encoder | Fused change | Encoder peak |
| ----: | ------------: | -----------------: | -----------: | -----------: |
|     1 |      3.797 ms |           4.337 ms |       +14.2% |    193.9 MiB |
|     4 |      7.914 ms |           8.406 ms |        +6.2% |    775.4 MiB |
|    16 |     26.937 ms |          25.249 ms |        -6.3% |  3,101.8 MiB |
|    32 |     53.550 ms |          49.294 ms |        -7.9% |  6,203.5 MiB |

The fused selector reduces selector workspace by approximately 97%. Full-model peak memory is the
same for both backends because transformer activations dominate the allocation.

See the package [README](https://github.com/NVlabs/WarpConvNet/tree/main/warpconvnet/models/uni3d)
for backend behavior and detailed constraints.
