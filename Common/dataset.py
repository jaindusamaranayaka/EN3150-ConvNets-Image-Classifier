from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# -------------------------------------------------
# Image transformations
# -------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -------------------------------------------------
# Dataset class
# -------------------------------------------------

class GTSRBDataset(Dataset):

    def __init__(
        self,
        dataframe,
        data_root,
        transform=None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        listed_path = self.data_root / row["Path"]
        image_path = listed_path
        if not image_path.exists():
            # A split CSV may use the extracted Kaggle layout
            # (Train/18/00018_00023_00028.png) or the original GTSRB layout
            # (Training/00018/00023_00028.ppm); look for the original file.
            path_parts = Path(row["Path"]).parts
            name_parts = Path(path_parts[-1]).stem.split("_")
            if len(path_parts) >= 3 and path_parts[-2].isdigit():
                class_id = int(path_parts[-2])
                # Kaggle names carry the class as an extra prefix
                if len(name_parts) == 3:
                    name_parts = name_parts[1:]
                filename = f"{'_'.join(name_parts)}.ppm"
                candidates = [
                    base / "Training" / f"{class_id:05d}" / filename
                    for base in (self.data_root, self.data_root / "GTSRB")
                ]
                image_path = next(
                    (candidate for candidate in candidates if candidate.exists()),
                    listed_path,
                )

        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in split was not found: {listed_path}")

        image = Image.open(
            image_path
        ).convert("RGB")

        label = int(
            row["ClassId"]
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -------------------------------------------------
# DataLoader creator
# -------------------------------------------------

def create_dataloaders(
    train_df,
    val_df,
    test_df,
    data_root,
    batch_size=32,
    num_workers=0,
    train_transform_override=None,
    eval_transform_override=None,
    seed=42
):

    # Fail early with a clear message instead of a missing-image error at the first batch
    if not Path(data_root).is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_root}. "
            "Run `python download_data.py` from the project root first."
        )

    selected_train_transform = (
        train_transform
        if train_transform_override is None
        else train_transform_override
    )

    selected_eval_transform = (
        eval_transform
        if eval_transform_override is None
        else eval_transform_override
    )

    train_dataset = GTSRBDataset(
        train_df,
        data_root,
        transform=selected_train_transform
    )

    val_dataset = GTSRBDataset(
        val_df,
        data_root,
        transform=selected_eval_transform
    )

    test_dataset = GTSRBDataset(
        test_df,
        data_root,
        transform=selected_eval_transform
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )
