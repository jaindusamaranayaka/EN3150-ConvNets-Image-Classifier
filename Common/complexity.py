import torch
import torch.nn as nn


# =========================================================
# Multiply-accumulate (MAC) counter
# =========================================================

def count_macs(
    model,
    input_size=(1, 3, 64, 64),
    per_layer=False
):
    """
    Count the multiply-accumulate operations for one forward pass.

    One MAC is one multiplication added to a running sum. Only
    Conv2d and Linear layers are counted, because they hold almost
    all of the work; ReLU, pooling, batch norm and bias additions
    are ignored (the usual convention).

    Conv2d: output elements x (input channels per group x kernel area)
            groups make this correct for depthwise convolutions.
    Linear: output elements x input features
    """

    layer_macs = []

    def conv_hook(module, inputs, output):

        kernel_ops = (
            module.in_channels
            // module.groups
            * module.kernel_size[0]
            * module.kernel_size[1]
        )

        layer_macs.append(
            (module, output.numel() // output.shape[0] * kernel_ops)
        )

    def linear_hook(module, inputs, output):

        layer_macs.append(
            (module, output.numel() // output.shape[0] * module.in_features)
        )

    hooks = []

    for module in model.modules():

        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))

        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    try:
        with torch.no_grad():
            model(torch.zeros(input_size, device=device))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    names = {
        module: name
        for name, module in model.named_modules()
    }

    total = sum(
        macs
        for _, macs in layer_macs
    )

    if per_layer:
        return total, [
            (names[module], macs)
            for module, macs in layer_macs
        ]

    return total
