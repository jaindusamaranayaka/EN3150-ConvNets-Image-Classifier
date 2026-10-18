import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# =========================================================
# Get predictions
# =========================================================

def get_predictions(
    model,
    loader,
    device
):

    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_true.extend(
                labels.numpy()
            )

            all_pred.extend(
                predictions
                .cpu()
                .numpy()
            )

    return (
        np.array(all_true),
        np.array(all_pred)
    )


# =========================================================
# Classification metrics
# =========================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "Test Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall
    }


# =========================================================
# Training curves
# =========================================================

def plot_training_curves(
    history,
    model_name,
    save_dir=None
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    # Loss
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{model_name} - Loss"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    if save_dir is not None:

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_dir
            / f"{model_name.lower().replace(' ', '_')}_loss.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    # Accuracy
    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"{model_name} - Accuracy"
    )

    plt.legend()
    plt.grid()

    plt.tight_layout()

    if save_dir is not None:

        plt.savefig(
            save_dir
            / f"{model_name.lower().replace(' ', '_')}_accuracy.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# =========================================================
# Confusion matrix
# =========================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name,
    save_dir=None
):

    cm = confusion_matrix(
        y_true,
        y_pred,
        normalize="true"
    )

    fig, ax = plt.subplots(
        figsize=(14, 14)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    display.plot(
        ax=ax,
        include_values=False,
        xticks_rotation=90,
        colorbar=True
    )

    plt.title(
        f"{model_name} - Normalized Confusion Matrix"
    )

    plt.tight_layout()

    if save_dir is not None:

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_dir
            / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()


# =========================================================
# Classification report
# =========================================================

def print_classification_report(
    y_true,
    y_pred
):

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    )


# =========================================================
# Parameter count
# =========================================================

def total_parameters(model):

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


def trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


# =========================================================
# Estimated FP32 memory
# =========================================================

def estimated_fp32_size_kb(model):

    parameters = total_parameters(
        model
    )

    return (
        parameters
        * 4
        / 1024
    )


# =========================================================
# Save model and measure actual file size
# =========================================================

def save_model_and_get_size(
    model,
    path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        path
    )

    size_kb = (
        os.path.getsize(path)
        / 1024
    )

    return size_kb


# =========================================================
# Inference time
# =========================================================

def measure_inference_time(
    model,
    loader,
    device,
    max_batches=50
):

    model.eval()

    total_time = 0.0
    total_images = 0

    # Warm-up
    with torch.no_grad():

        for images, _ in loader:

            images = images.to(device)

            for _ in range(3):
                _ = model(images)

            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():

        for batch_index, (
            images,
            _
        ) in enumerate(loader):

            if batch_index >= max_batches:
                break

            images = images.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            total_time += elapsed
            total_images += images.size(0)

    ms_per_image = (
        total_time
        / total_images
        * 1000
    )

    return ms_per_image