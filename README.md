# EN3150 Assignment 03: Resource-Constrained CNN for Edge Image Classification

Traffic-sign classification on **GTSRB** (43 classes) at **64 × 64** input, comparing two custom CNNs designed for
edge devices with two fine-tuned lightweight pretrained networks.

| Model | Type | Parameters | Test accuracy |
|---|---|---|---|
| Model A | Standard CNN (Conv2d + max-pooling) | 45,643 | 98.97% |
| Model B | Depthwise separable CNN (< 100k parameters) | 25,526 | 97.00% |
| MobileNetV2 | ImageNet-pretrained, fine-tuned | 2,278,955 | 100.00% |
| EfficientNet-B0 | ImageNet-pretrained, fine-tuned | 4,062,631 | 99.82% |

Implemented in Python with PyTorch.

## Setup

Requires Python 3.10+ (developed with 3.11). A CUDA GPU is optional but makes training much faster.

```bash
pip install -r requirements.txt
python download_data.py
```

`download_data.py` downloads the GTSRB training set (about 180 MB) to `data/gtsrb/` and checks that every image listed
in `splits/` exists. `data/`, `checkpoints/` and model weights are not committed to git.

On Windows, `pyzmq` older than 27 can stop `import torch` from working inside Jupyter (`WinError 1114`);
`requirements.txt` therefore requires `pyzmq>=27`.

## Dataset and split

- **GTSRB**: German Traffic Sign Recognition Benchmark, 26,640 images in 43 classes.
- Images are resized to 64 × 64 to emulate a low-resolution edge sensor.
- One fixed, disjoint split shared by every model, stored in `splits/`: **70% train (18,648) / 15% validation
  (3,996) / 15% test (3,996)**, seed 42.
- Training images are augmented (colour jitter, small rotation, shift and scale); validation and test images are not.

## Repository structure

```
Common/                     Shared code used by all models
  dataset.py                  Split CSV -> images -> batches (transforms, augmentation, data loaders)
  models.py                   Model A, Model B and the depthwise separable convolution block
  training.py                 Training / evaluation loops, optimizer factory
  evaluation.py               Metrics, plots, model size and inference time
  complexity.py               Multiply-accumulate (MAC) counter
splits/                     train / val / test CSVs (class id and image path)
Optimizer/
  Main_Study_ModelA/          Task 3: SGD vs SGD + Momentum (beta sweep) vs Adam
  Preliminary_Run_ModelB/     Early comparison on Model B; writes results/Optimizer/selected_optimizer.txt
Model_A/src/ModelA.ipynb    Task 2 and 4: standard CNN
Model_B/src/ModelB.ipynb    Task 2 and 4: depthwise separable CNN
SOTA/MobileNetV2/           Task 5: MobileNetV2 fine-tuning
SOTA/EfficientNetB0/        Task 5: EfficientNet-B0 fine-tuning
Comparison/src/
  ModelA_vs_ModelB.ipynb      Task 4 comparison table and figures
  ModelB_vs_SOTA.ipynb        Task 6 comparison table and trade-off plots
results/                    All saved outputs, one folder per model plus Optimizer, Dataset and Comparisons
Report/
  DINO_A03_EN3150.tex         LaTeX source of the report
  figures/                    Copies of the result figures used in the report
  build/DINO_A03_EN3150.pdf   Compiled report
Documents/                  Assignment brief
download_data.py            Dataset download and check
```

## Report

The report is `Report/DINO_A03_EN3150.tex`; the compiled PDF is `Report/build/DINO_A03_EN3150.pdf`. To rebuild it, run
`pdflatex` twice from the `Report` folder (the second run fills in the contents and cross-references):

```bash
cd Report
pdflatex -output-directory=build DINO_A03_EN3150.tex
pdflatex -output-directory=build DINO_A03_EN3150.tex
```

The folder is self-contained, so it can also be uploaded to Overleaf as it is. The figures in `Report/figures/` are
copies of those in `results/`; copy them again if a model is retrained.

## Running the notebooks

Open the notebooks from the repository (each one finds the project root itself) and run all cells, in this order:

| Step | Notebook | Needs | Time (GTX 1660 Ti) |
|---|---|---|---|
| 1 | `Optimizer/Main_Study_ModelA/src/optimizer_benchmark.ipynb` | dataset (uses its own seeded split, not `splits/`) | ~50 min |
| 2 | `Model_A/src/ModelA.ipynb` | dataset, `results/Optimizer/selected_optimizer.txt` | ~20 min |
| 3 | `Model_B/src/ModelB.ipynb` | dataset, `results/Optimizer/selected_optimizer.txt` | ~20 min |
| 4 | `SOTA/MobileNetV2/src/MobileNetV2_Retrain.ipynb` | dataset, internet (ImageNet weights) | ~20 min |
| 5 | `SOTA/EfficientNetB0/src/EfficientNetB0_Retrain.ipynb` | dataset, internet (ImageNet weights) | ~25 min |
| 6 | `Comparison/src/ModelA_vs_ModelB.ipynb` | results of steps 2-3 | seconds |
| 7 | `Comparison/src/ModelB_vs_SOTA.ipynb` | results of steps 2-5 | ~1 min |

- `selected_optimizer.txt` (Adam) is already committed, so steps 2-7 can run without step 1. The optimizer study
  (step 1) writes it from its validation results; the earlier `Optimizer/Preliminary_Run_ModelB/` notebook does too.
- All notebooks fix their random seeds (42) and use deterministic cuDNN settings.
- Run the training notebooks one at a time on an otherwise idle computer: they also measure training time per epoch.

## Where each task's results are

| Task | Outputs |
|---|---|
| 1. Data preparation | `splits/`, `results/Dataset/class_distribution.png`, split checks in the model notebooks |
| 2. Architectures | Layer and parameter tables in `Model_A` / `Model_B` notebooks; MACs via `Common/complexity.py` |
| 3. Optimizer selection | `Optimizer/Main_Study_ModelA/src/outputs_member3/` (summary table, convergence, momentum sweep), conclusions at the end of the notebook |
| 4. Training and evaluation | `results/Model_A/`, `results/Model_B/` (loss curves, confusion matrices, metrics); `results/Comparisons/model_a_vs_model_b.*` |
| 5. SOTA fine-tuning | `results/MobileNetV2/`, `results/EfficientNetB0/` |
| 6. Final comparison | `results/Comparisons/final_comparison.csv`, `final_tradeoffs.png`, `final_validation_accuracy.png` |

Each model folder in `results/` contains `*_metrics.csv`, `*_history.json` (per-epoch loss, accuracy and time),
`*_classification_report.json`, `*_confusion_matrix.csv` and the loss, accuracy and confusion-matrix figures.

## Key results

**Model A vs Model B** (Task 4, same pipeline, 25 epochs each):

| | Model A | Model B | Change |
|---|---|---|---|
| Parameters | 45,643 | 25,526 | -44.1% |
| MACs per image | 9.46 M | 1.42 M | -85.0% |
| Size on disk | 182.8 KB | 105.7 KB | -42.2% |
| Training time per epoch (median) | 41.0 s | 44.2 s | +7.8% |
| Test accuracy | 98.97% | 97.00% | -1.98 pts |

**Model B vs lightweight SOTA** (Task 6):

| | Model B | MobileNetV2 | EfficientNet-B0 |
|---|---|---|---|
| Test accuracy | 97.00% | 100.00% | 99.82% |
| Size on disk | 0.10 MB | 8.93 MB | 15.79 MB |
| MACs per image | 1.42 M | 24.50 M | 32.02 M |
| CPU latency (1 thread, batch 1) | 1.61 ms | 6.94 ms | 10.35 ms |

CPU latency depends on the machine and varies slightly between runs; the ratios between models are what matter.
Training time per epoch is reported as the median over all epochs, so a single epoch slowed by other activity on
the computer does not distort it.
