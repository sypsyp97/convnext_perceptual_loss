# ConvNext Perceptual Loss

## Motivation

Traditional pixel-wise loss functions like MSE or L1 often fail to capture the perceptual quality of images, leading to blurry or unrealistic results in many computer vision tasks. While [VGG-based perceptual losses](https://arxiv.org/abs/1603.08155) have been widely used to address this issue, they rely on older architecture designs that may not capture image features as effectively as modern models.

This package introduces a PyTorch perceptual loss implementation based on the [ConvNext](https://arxiv.org/abs/2201.03545) architecture. These models have shown superior performance in various vision tasks, making them excellent feature extractors for perceptual loss computation.

## Features

- Support for different ConvNext model scales (`TINY`, `SMALL`, `BASE`, `LARGE`)
- Configurable feature layers and weights for fine-grained control
- Optional Gram matrix computation for style transfer tasks

## Installation

```bash
git clone https://github.com/sypsyp97/convnext_perceptual_loss.git
cd convnext_perceptual_loss
pip install -e .
```

## Usage

```python
import torch
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType

# Initialize the loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = ConvNextPerceptualLoss(
    model_type=ConvNextType.TINY,
    device=device,
    feature_layers=[0, 2, 4, 6, 8, 10, 12, 14],
    use_gram=False,
    layer_weight_decay=0.99 
)

# Example 1: Computing loss between RGB images
rgb_generated = torch.randn(1, 3, 256, 256).to(device)  # Generated/predicted image
rgb_target = torch.randn(1, 3, 256, 256).to(device)     # Ground truth image
perceptual_loss_rgb = loss_fn(rgb_generated, rgb_target)
print(f"RGB Perceptual Loss: {perceptual_loss_rgb.item():.4f}")

# Example 2: Computing loss between grayscale images
gray_generated = torch.randn(1, 1, 256, 256).to(device)
gray_target = torch.randn(1, 1, 256, 256).to(device)
perceptual_loss_gray = loss_fn(gray_generated, gray_target)
print(f"Grayscale Perceptual Loss: {perceptual_loss_gray.item():.4f}")
```

The loss function automatically handles both RGB (3-channel) and grayscale (1-channel) images. Input tensors should follow the PyTorch convention of `(batch_size, channels, height, width)` format.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
