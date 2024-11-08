import torch
import torch.nn as nn
import torchvision.models
from enum import Enum
from typing import List, Tuple, Optional


class ConvNextType(Enum):
    """Available ConvNext model types"""
    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"

def make_dummy_data(batch_size=1, channels=3, height=256, width=256, device='cuda'):
    """Create random tensor data for testing"""
    return torch.randn(batch_size, channels, height, width, device=device)

class ConvNextPerceptualLoss(nn.Module):
    """ConvNext Perceptual Loss Module"""
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
        """Initialize perceptual loss module"""
        super().__init__()
        
        self.device = device
        self.input_range = input_range
        self.use_gram = use_gram
        self.feature_layers = feature_layers
        
        # Calculate weights with decay if not specified
        if feature_weights is None:
            decay_values = [layer_weight_decay ** i for i in range(len(feature_layers))]
            weights = torch.tensor(decay_values, device=device, dtype=torch.float32)
            weights = weights / weights.sum()
        else:
            weights = torch.tensor(feature_weights, device=device, dtype=torch.float32)
        
        assert len(feature_layers) == len(weights), "Number of feature layers must match number of weights"
        self.register_buffer("feature_weights", weights)
        
        # Load pretrained ConvNext model
        model_name = f"convnext_{model_type.value}"
        try:
            weights_enum = getattr(torchvision.models, f"ConvNeXt_{model_type.value.capitalize()}_Weights")
            weights = weights_enum.DEFAULT
            model = getattr(torchvision.models, model_name)(weights=weights)
        except (AttributeError, ImportError):
            model = getattr(torchvision.models, model_name)(pretrained=True)
        
        # Extract blocks and ensure they're in eval mode
        self.blocks = nn.ModuleList()
        for stage in model.features:
            if isinstance(stage, nn.Sequential):
                self.blocks.extend(stage)
            else:
                self.blocks.append(stage)
        
        self.blocks = self.blocks.eval().to(device)
        # Don't freeze parameters but set requires_grad=False since we don't update them
        for param in self.blocks.parameters():
            param.requires_grad_(False)
        
        # Register normalization parameters
        self.register_buffer(
            "mean", 
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", 
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        )
        
        self.to(device)

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor"""
        x = x.to(self.device)
        
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Ensure we create new leaf tensors while maintaining gradient flow
        x = x - torch.tensor(0., device=self.device)  # Create new leaf tensor
        
        min_val, max_val = self.input_range
        x = (x - min_val) / (max_val - min_val)
        x = (x - self.mean) / self.std
        
        if x.requires_grad:
            x.retain_grad()  # Retain gradients for intermediate values
            
        return x

    def gram_matrix(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute Gram matrix of feature maps"""
        b, c, h, w = x.size()
        features = x.view(b, c, -1)
        gram = torch.bmm(features, features.transpose(1, 2))
        if normalize:
            gram = gram / (c * h * w)
        return gram
    
    def compute_feature_loss(
        self, 
        input_features: List[torch.Tensor],
        target_features: List[torch.Tensor],
        layers_indices: List[int],
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature loss ensuring scalar output"""
        losses = []
        
        for idx, weight in zip(layers_indices, weights):
            input_feat = input_features[idx]
            target_feat = target_features[idx].detach()  # Detach target features
            
            if self.use_gram:
                input_gram = self.gram_matrix(input_feat)
                target_gram = self.gram_matrix(target_feat)
                layer_loss = nn.functional.l1_loss(input_gram, target_gram)
            else:
                layer_loss = nn.functional.mse_loss(input_feat, target_feat)
            
            losses.append(weight * layer_loss)
            
        # Sum all losses and ensure scalar output
        return torch.stack(losses).sum()

    def forward(
        self, 
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to compute loss"""
        input = input.to(self.device)
        target = target.to(self.device)
        
        input = self.normalize_input(input)
        target = self.normalize_input(target)
        
        # Extract features
        input_features = []
        target_features = []
        
        x_input = input
        x_target = target
        for block in self.blocks:
            x_input = block(x_input)
            with torch.no_grad():  # No need to compute gradients for target features
                x_target = block(x_target)
            input_features.append(x_input)
            target_features.append(x_target)
        
        loss = self.compute_feature_loss(
            input_features, target_features,
            self.feature_layers, self.feature_weights
        )
        
        return loss

def main():
    """Main function for testing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    loss_fn = ConvNextPerceptualLoss(
        model_type=ConvNextType.TINY,
        device=device,
        feature_layers=[0, 2, 4],
        use_gram=False,
        layer_weight_decay=0.9
    )
    
    # Test with small images for quick verification
    input_image = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    target_image = torch.randn(2, 3, 64, 64, device=device)
    
    loss = loss_fn(input_image, target_image)
    print(f"Loss shape: {loss.shape}, Loss dim: {loss.dim()}")
    loss.backward()
    
    print(f"Loss: {loss.item()}")
    print(f"Input grad exists: {input_image.grad is not None}")
    if input_image.grad is not None:
        print(f"Grad norm: {input_image.grad.norm().item()}")


if __name__ == '__main__':
    main()