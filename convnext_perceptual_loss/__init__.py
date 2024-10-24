# convnext_perceptual_loss/__init__.py
from .loss import (
    ConvNextPerceptualLoss,
    ConvNextType,
    make_dummy_data
)

__version__ = "0.1.0"
__all__ = [
    "ConvNextPerceptualLoss",
    "ConvNextType",
    "make_dummy_data"
]