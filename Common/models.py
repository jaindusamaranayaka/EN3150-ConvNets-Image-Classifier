import torch.nn as nn


# =========================================================
# Model A - Standard CNN
# =========================================================

class ModelA(nn.Module):

    def __init__(self, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# =========================================================
# Depthwise Separable Convolution
# =========================================================

class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)

        return x


# =========================================================
# Model B - Lightweight CNN
# =========================================================

class ModelB(nn.Module):

    def __init__(self, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(

            DepthwiseSeparableConv(
                3, 16
            ),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(
                16, 32
            ),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(
                32, 32
            ),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(
                32, 32
            ),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# =========================================================
# Parameter counter
# =========================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
