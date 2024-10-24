# ConvNext Perceptual Loss

A PyTorch implementation of perceptual loss using ConvNext models. This package provides a flexible and efficient way to compute perceptual losses for various computer vision tasks such as style transfer, super-resolution, and image-to-image translation.

## Features

- Support for different ConvNext model scales (TINY, SMALL, BASE, LARGE)
- Configurable feature layers and weights
- Optional Gram matrix computation for style transfer
- Automatic input normalization
- GPU support
- Customizable layer weight decay

## Installation


### From source

```bash
git clone https://github.com/sypsyp97/ConvNext_Perceptual_Loss.git
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
    device=device,
    model_type=ConvNextType.TINY,
    feature_layers=[0, 2, 4, 6],
    use_gram=True
)

# Your input and target tensors (B, C, H, W)
input_image = torch.randn(1, 3, 256, 256).to(device)
target_image = torch.randn(1, 3, 256, 256).to(device)

# Compute loss
loss = loss_fn(input_image, target_image)
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
