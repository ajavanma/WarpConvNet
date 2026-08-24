# PPAT

PPAT (Point Patch Transformer) maps a colored point cloud into an OpenShape CLIP embedding. Shapes
can then be ranked against text labels without training a task-specific classifier. WarpConvNet
loads the released OpenShape PointBERT checkpoints directly.

PPAT is separate from [Uni3D](uni3d.md). Each family requires text features from its own paired
text encoder; raw cosine scores are not comparable across the two embedding spaces.

## Supported models

| Variant                          | Up axis | Output width | Text space       |
| -------------------------------- | ------- | -----------: | ---------------- |
| `openshape-pointbert-vitb32-rgb` | Y       |          512 | CLIP ViT-B/32    |
| `openshape-pointbert-vitl14-rgb` | Y       |          768 | CLIP ViT-L/14    |
| `openshape-pointbert-vitg14-rgb` | Z       |        1,280 | CLIP ViT-bigG/14 |

All variants accept 10,000-point XYZ+RGB inputs and load their matching OpenShape weights. B32 and
L14 are Y-up; bigG is Z-up. Rotate the shape into the checkpoint's up-axis convention before
normalization. The model and `normalize_point_cloud` do not rotate coordinates.

## Usage

Install the shared demo dependencies for Hugging Face downloads and mesh loading:

```bash
pip install -e ".[demo]"
```

```python
import torch

from warpconvnet.models.ppat import load_openshape_pointbert, normalize_point_cloud

model = load_openshape_pointbert(
    "openshape-pointbert-vitg14-rgb",
    device="cuda",
)

# xyz and rgb have shape (batch, points, 3). xyz is already z-up.
xyz = normalize_point_cloud(xyz)
points = torch.cat((xyz, rgb), dim=-1).permute(0, 2, 1).contiguous()

with torch.inference_mode():
    embedding = model(points.cuda())
```

The model input has shape `(batch, 6, points)`, with XYZ in channels 0-2 and RGB in channels 3-5.
RGB values are in `[0, 1]`. `normalize_point_cloud` centers XYZ and scales it to the unit ball; it
does not rotate the input.

`load_openshape_pointbert` downloads and strictly loads published weights. Set `checkpoint_path`
for local weights or use `build_openshape_pointbert` to create an uninitialized model.

The bundled mesh demo downloads the bigG model and prompt-averaged LVIS text features:

```bash
python examples/ppat_demo.py
```

It prints labels ranked by cosine similarity. These values are relative scores, not probabilities.

## Neighborhood backends

| Backend                       | Device   | Behavior                                                    |
| ----------------------------- | -------- | ----------------------------------------------------------- |
| `cumsum`                      | CPU/CUDA | Reference-compatible point order and default                |
| `dense`                       | CPU/CUDA | Dense reference implementation; bit-identical to `cumsum`   |
| `warp`                        | CUDA     | Exact cell-list radius search; lower memory but not faster  |
| `ball_cuda`                   | CUDA     | Exact spherical rule with direct-distance boundary rounding |
| `voxel:SIZE`                  | CPU/CUDA | Approximate 3x3x3 voxel neighborhood                        |
| `voxel_cuda:SIZE`             | CUDA     | Fused approximate voxel neighborhood                        |
| `l1`, `linf`, `random`, `knn` | CPU/CUDA | Optional approximate selectors                              |

`warp` and `ball_cuda` follow the same radius rule as the reference path but compute distance
directly. Floating-point rounding at the radius boundary can therefore differ from `cumsum`.
Approximate backends can change model accuracy and should use the same normalization and
configuration during evaluation and deployment.

## Accuracy and performance

The [OpenShape checkpoint table](https://github.com/Colin97/OpenShape_code#checkpoints) reports
46.8% Objaverse-LVIS zero-shot top-1 accuracy for `pointbert-vitg14-rgb`.

The latency measurements below use the supported WarpConvNet path in BF16 on an NVIDIA RTX PRO
6000 Blackwell (`sm_120`).

| Batch 32, 10,000 points |  Latency |
| ----------------------- | -------: |
| Whole model             | 17.06 ms |
| Transformer             |  4.69 ms |

| Ball query, B=96, N=10,000, S=384, K=64 |     Time | Peak memory |
| --------------------------------------- | -------: | ----------: |
| `voxel_cuda:0.1`                        |  0.53 ms |    0.10 GiB |
| `ball_cuda`                             |  0.79 ms |    0.10 GiB |
| `cumsum`                                | 20.25 ms |    3.09 GiB |
| `dense`                                 | 25.30 ms |    9.63 GiB |

`voxel_cuda:0.1` is approximate. `ball_cuda`, `cumsum`, and `dense` implement the same spherical
neighborhood rule, subject to the floating-point boundary behavior described above.

See the package [README](https://github.com/NVlabs/WarpConvNet/tree/main/warpconvnet/models/ppat)
for the full input contract and demo options.
