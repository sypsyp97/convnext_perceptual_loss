import torch
import torch.nn as nn
import torchvision.models
from enum import Enum
from typing import List, Tuple, Optional


class ConvNextType(Enum):
    """Available ConvNext model types
    
    Contains four different model scales:
    - TINY: Smallest model, fewer parameters, fast inference
    - SMALL: Small model, balances performance and speed
    - BASE: Default configuration
    - LARGE: Largest model, best performance but computationally intensive
    """
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"

def make_dummy_data(batch_size=1, channels=3, height=256, width=256, device='cuda'):
    """Create random tensor data for testing
    
    Args:
        batch_size (int): Size of the batch
        channels (int): Number of channels, default is 3 (RGB image)
        height (int): Image height
        width (int): Image width
        device (str): Computing device, 'cuda' or 'cpu'
    
    Returns:
        torch.Tensor: Random tensor of shape (batch_size, channels, height, width)
    """
    return torch.randn(batch_size, channels, height, width, device=device)

class ConvNextPerceptualLoss(nn.Module):
    """ConvNext Perceptual Loss Module
    
    This module uses a pretrained ConvNext model to extract features and compute
    the feature differences between input and target images. It can optionally use
    Gram matrices to capture style features or directly compare feature maps.
    
    Key features:
    1. Supports multi-layer feature extraction and weighted combination
    2. Configurable weight decay mechanism
    3. Flexible loss computation (Gram matrix or direct feature matching)
    4. Automatic input normalization
    """
    def __init__(
        self, 
        device: torch.device,
        model_type: ConvNextType = ConvNextType.TINY,
        feature_layers: List[int] = [0, 2, 4, 6, 8, 10, 12, 14],
        feature_weights: Optional[List[float]] = None,
        use_gram: bool = True,
        input_range: Tuple[float, float] = (-1, 1),
        layer_weight_decay: float = 1.0
    ):
        """Initialize the perceptual loss module
        
        Args:
            device (torch.device): Computing device
            model_type (ConvNextType): Type of ConvNext model
            feature_layers (List[int]): List of layer indices for feature extraction
            feature_weights (Optional[List[float]]): Weights for each feature layer, auto-calculated if None
            use_gram (bool): Whether to use Gram matrix for loss computation
            input_range (Tuple[float, float]): Range of input image values
            layer_weight_decay (float): Weight decay factor for automatic weight calculation
        """
        super().__init__()
        
        self.device = device
        self.input_range = input_range
        self.use_gram = use_gram
        
        # Calculate weights with decay if not specified
        if feature_weights is None:
            # Calculate weights using exponential decay based on layer depth
            feature_weights = [layer_weight_decay ** i for i in range(len(feature_layers))]
            # Normalize weights to sum to 1
            total_weight = sum(feature_weights)
            feature_weights = [w / total_weight for w in feature_weights]
            
        assert len(feature_layers) == len(feature_weights), "Number of feature layers must match number of weights"
        
        self.feature_layers = feature_layers
        self.feature_weights = feature_weights
        
        # Load pretrained ConvNext model
        model_name = f"convnext_{model_type.value}"
        try:
            # Try new-style weight loading
            weights_enum = getattr(torchvision.models, f"ConvNeXt_{model_type.value.capitalize()}_Weights")
            weights = weights_enum.DEFAULT
            model = getattr(torchvision.models, model_name)(weights=weights)
        except (AttributeError, ImportError):
            # Fallback to old-style weight loading
            model = getattr(torchvision.models, model_name)(pretrained=True)
        
        # Extract all convolutional blocks
        self.blocks = nn.ModuleList()
        for stage in model.features:
            if isinstance(stage, nn.Sequential):
                self.blocks.extend(stage)
            else:
                self.blocks.append(stage)
                
        # Move model to device and set to evaluation mode
        self.blocks = self.blocks.eval().to(self.device)
        # Freeze model parameters
        self.blocks.requires_grad_(False)
        
        # Register ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor
        
        Performs the following steps:
        1. Moves input to correct device
        2. Converts single-channel images to three channels
        3. Normalizes pixel values to [0,1]
        4. Applies ImageNet normalization
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Normalized tensor
        """
        x = x.to(self.device)
        
        # Convert single-channel images to three channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Range normalization
        min_val, max_val = self.input_range
        x = (x - min_val) / (max_val - min_val)
        # ImageNet normalization
        x = (x - self.mean) / self.std
        return x

    def gram_matrix(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute Gram matrix of feature maps
        
        The Gram matrix represents correlations between different channels
        of the feature map, commonly used to capture style features.
        
        Args:
            x (torch.Tensor): Input feature map, shape [B, C, H, W]
            normalize (bool): Whether to normalize the Gram matrix
            
        Returns:
            torch.Tensor: Gram matrix, shape [B, C, C]
        """
        b, c, h, w = x.size()
        # Reshape features to [B, C, H*W]
        features = x.view(b, c, -1)
        # Compute channel correlations
        gram = torch.bmm(features, features.transpose(1, 2))
        # Normalize
        if normalize:
            gram = gram / (c * h * w)
        return gram
    
    def compute_feature_loss(
        self, 
        input_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
        layers_indices: List[int],
        weights: List[float]
    ) -> torch.Tensor:
        """Compute feature loss
        
        Calculates weighted loss for specified layers using either
        Gram matrix or direct feature matching.
        
        Args:
            input_features: List of input image features
            target_features: List of target image features
            layers_indices: Indices of layers to compute loss for
            weights: Weights for each layer
            
        Returns:
            torch.Tensor: Weighted feature loss
        """
        total_loss = 0.0
        
        for idx, weight in zip(layers_indices, weights):
            input_feat = input_features[idx].float()
            target_feat = target_features[idx].float()
            
            if self.use_gram:
                # Compute style loss using Gram matrix
                input_feat = self.gram_matrix(input_feat)
                target_feat = self.gram_matrix(target_feat)
                # Use L1 loss for Gram matrices
                loss = torch.nn.functional.l1_loss(input_feat, target_feat)
            else:
                # Use L2 loss for direct feature matching
                loss = torch.nn.functional.mse_loss(input_feat, target_feat)
            
            total_loss += weight * loss
            
        return total_loss

    def forward(
        self, 
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to compute loss
        
        Steps:
        1. Normalize input and target images
        2. Extract features through ConvNext model
        3. Compute feature loss
        
        Args:
            input (torch.Tensor): Input image
            target (torch.Tensor): Target image
            
        Returns:
            torch.Tensor: Computed perceptual loss
        """
        # Normalize inputs
        input = self.normalize_input(input)
        target = self.normalize_input(target)
        
        # Collect features from each layer
        input_features = []
        target_features = []
        
        x, y = input, target
        
        # Extract features layer by layer
        with torch.no_grad():
            for block in self.blocks:
                x = block(x)
                y = block(y)
                input_features.append(x)
                target_features.append(y)
        
        # Compute weighted feature loss
        loss = self.compute_feature_loss(
            input_features, target_features,
            self.feature_layers, self.feature_weights
        )
        
        return loss

def main():
    """Main function for testing the perceptual loss module
    
    Creates random test data and computes loss to verify module functionality.
    """

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    
    # Set computing device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create loss function instance
    loss_fn = ConvNextPerceptualLoss(
        model_type=ConvNextType.TINY,
        device=device,
        feature_layers=[0, 2, 4, 6, 8, 10, 12, 14],  # Use multiple layer features
        use_gram=False,  # Use Gram matrix for loss computation
        layer_weight_decay=0.9  # Set weight decay factor
    )
    
    # Create test data
    batch_size = 1
    channels = 3
    height = 512
    width = 512
    
    # Generate random input and target images
    input_image = make_dummy_data(batch_size, channels, height, width, device=device)
    target_image = make_dummy_data(batch_size, channels, height, width, device=device)
    
    # Compute and print loss
    loss = loss_fn(input_image, target_image)
    print(f"Computed loss: {loss.item()}")

if __name__ == '__main__':
    main()