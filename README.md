# EN3150 ConvNets Image Classifier

Traffic sign classification experiments for the EN3150 assignment. The project compares two custom CNNs, optimizer choices, and pretrained SOTA architectures.

## Repository layout

| Path | Contents |
| --- | --- |
| `Common/` | Shared datasets, model definitions, training, and evaluation helpers |
| `Model_A/src/` | Model A notebook, plots, and report exports |
| `Model_A/src/optimizer/` | Main controlled optimizer benchmark and saved outputs |
| `Model_B/src/` | Model B notebook and report exports |
| `SOTA/MobileNetV2/src/` | Shared-recipe `MobileNetV2_Retrain.ipynb`, Member 3 `mobilenetv2_finetune.ipynb`, and Member 3 saved artifacts in `outputs_member3/` |
| `SOTA/EfficientNetB0/src/` | Trained `EfficientNetB0_Retrain.ipynb` |
| `results/figures/` | Shared plots for model comparisons |
| `results/metrics/` | Model metrics, classification report, and confusion-matrix data |
| `results/tables/` | Model A/B and final model comparison tables |
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
- The shared-recipe MobileNetV2 and EfficientNetB0 retraining notebooks use the project splits and have corresponding metrics in `results/metrics/`.
- `results/FinalComparison.ipynb` and `results/tables/final_model_comparison.csv` compare Model B with MobileNetV2 and EfficientNetB0 for Task 6.
- `Model_A/src/optimizer/optimizer_benchmark.ipynb` is the controlled optimizer study; it compares Adam, SGD, and SGD with momentum and includes a momentum sweep. Its saved summary and plots are in `Model_A/src/optimizer/outputs_member3/`. When rerun, it selects an optimizer from validation metrics and writes `results/metrics/selected_optimizer.txt` for the Model A and Model B training notebooks.

The notebooks and result files provide evidence for the modeling, optimizer, evaluation, and SOTA comparison tasks. A separate final assignment report PDF was not found in this repository during review.

Large datasets, Python caches, and model checkpoints are excluded from Git. Recreate model weights by running the relevant training notebook or script.
