# EN3150 ConvNets Image Classifier

Traffic sign classification experiments for the EN3150 assignment. The project compares two custom CNNs, optimizer choices, and pretrained SOTA architectures.

## Repository layout

| Path | Contents |
| --- | --- |
| `Common/` | Shared datasets, model definitions, training, and evaluation helpers |
| `Model_A/src/` | Model A notebook, training script, plots, and report exports |
| `Model_B/src/` | Model B notebook and LaTeX export |
| `Optimizer_Comparison/src/` | Optimizer comparison notebook and LaTeX export |
| `Optimizer_Benchmark/src/` | Standalone optimizer benchmark and saved outputs |
| `SOTA/MobileNetV2/src/` | Trained MobileNetV2 notebook and saved artifacts in `outputs_member3/` |
| `SOTA/EfficientNetB0/src/` | EfficientNetB0 workflow notebook (run outputs are not saved here yet) |
| `results/figures/` | Shared plots for model and optimizer comparisons |
| `results/metrics/` | Model metrics, classification report, and confusion-matrix data |
| `results/tables/` | Summary comparison tables |
| `splits/` | Train, validation, and test split manifests |
| `data/` | Local GTSRB dataset (ignored by Git) |
| `checkpoints/` | Local model and pretrained checkpoints (ignored by Git) |

## Setup

Install the packages listed in `requirements.txt`, then open notebooks from the repository. Notebooks locate the project root from `.git` and expect the dataset under `data/gtsrb/` and split manifests under `splits/`.

```powershell
python -m pip install -r requirements.txt
```

## Saved results

- Model A has executed training and evaluation output in its notebook and plots under `Model_A/src/`.
- Model B's metrics and plots are in `results/`; its notebook includes the matching saved evaluation outputs.
- Fine-tuned MobileNetV2 results are in `SOTA/MobileNetV2/src/outputs_member3/`.
- `SOTA/MobileNetV2` and `SOTA/EfficientNetB0` currently contain workflows without saved run artifacts.

Large datasets, Python caches, and model checkpoints are excluded from Git. Recreate model weights by running the relevant training notebook or script.
