import time
import copy

import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


def evaluate_epoch(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device
):

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "epoch_time": []
    }

    best_val_loss = float("inf")

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(epochs):

        start_time = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_acc
        )

        history["epoch_time"].append(
            elapsed
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_weights = copy.deepcopy(
                model.state_dict()
            )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.2f}s"
        )

    model.load_state_dict(
        best_weights
    )

    return model, history


import torch.optim as optim


def create_optimizer(
    optimizer_name,
    parameters
):

    if optimizer_name == "Adam":

        return optim.Adam(
            parameters,
            lr=1e-3
        )

    elif optimizer_name == "SGD":

        return optim.SGD(
            parameters,
            lr=1e-2
        )

    elif optimizer_name == "SGD_Momentum":

        return optim.SGD(
            parameters,
            lr=1e-2,
            momentum=0.9
        )

    else:

        raise ValueError(
            f"Unknown optimizer: "
            f"{optimizer_name}"
        )