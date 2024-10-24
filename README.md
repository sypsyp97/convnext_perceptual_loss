# ConvNext Perceptual Loss

## Motivation

Traditional pixel-wise loss functions like MSE or L1 often fail to capture the perceptual quality of images, leading to blurry or unrealistic results in many computer vision tasks. While VGG-based perceptual losses have been widely used to address this issue, they rely on older architecture designs that may not capture modern image features effectively.

This package introduces a perceptual loss implementation based on the modern ConvNext architecture. ConvNext models have shown superior performance in various vision tasks, making them excellent feature extractors for perceptual loss computation. The hierarchical feature representation and modern architectural improvements in ConvNext lead to better capture of both low-level details and high-level semantic information.

## Features

- Support for different ConvNext model scales (TINY, SMALL, BASE, LARGE)
- Configurable feature layers and weights for fine-grained control
- Optional Gram matrix computation for style transfer tasks
- Customizable layer weights or weight decay for balanced feature importance

## Installation

```bash
git clone https://github.com/sypsyp97/ConvNext_Perceptual_Loss.git
cd ConvNext_Perceptual_Loss
pip install -e .
```

## Usage

```python
import torch
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType

# Initialize the loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = ConvNextPerceptualLoss(
    device=device,
    model_type=ConvNextType.TINY,
    feature_layers=[0, 2, 4, 6],
    use_gram=False
)

# Example 1: RGB Images (B, C=3, H, W)
rgb_input = torch.randn(1, 3, 256, 256).to(device)
rgb_target = torch.randn(1, 3, 256, 256).to(device)
rgb_loss = loss_fn(rgb_input, rgb_target)

# Example 2: Grayscale Images (B, C=1, H, W)
gray_input = torch.randn(1, 1, 256, 256).to(device)
gray_target = torch.randn(1, 1, 256, 256).to(device)
gray_loss = loss_fn(gray_input, gray_target)
```

The loss function automatically handles both RGB (3-channel) and grayscale (1-channel) images. Input tensors should follow the PyTorch convention of `(batch_size, channels, height, width)` format.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
