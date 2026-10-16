import torch.nn as nn


# =========================================================
# Model A - Standard CNN
# =========================================================

class ModelA(nn.Module):

    def __init__(self, num_classes=43):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 160,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                160,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
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
                3, 32
            ),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(
                32, 64
            ),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(
                64, 128
            ),
            nn.MaxPool2d(2),

            DepthwiseSeparableConv(
                128, 160
            ),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                160,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
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