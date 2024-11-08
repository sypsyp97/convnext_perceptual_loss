import unittest
import torch
import torch.nn as nn
from torch.optim import Adam
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType

class ResBlock(nn.Module):
    """Residual block that maintains input dimensions"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Skip connection
        return self.relu(out)

class SimpleTransformNet(nn.Module):
    """A simple network that transforms images while maintaining dimensions"""
    def __init__(self, channels=3, base_filters=64):
        super().__init__()
        self.net = nn.Sequential(
            # Initial convolution
            nn.Conv2d(channels, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Residual blocks
            ResBlock(base_filters),
            ResBlock(base_filters),
            ResBlock(base_filters),
            
            # Final convolution to return to input channels
            nn.Conv2d(base_filters, channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class TestTrainingWithPerceptualLoss(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Create the networks
        self.transform_net = SimpleTransformNet(base_filters=32).to(self.device)
        
        # Create the perceptual loss
        self.perceptual_loss = ConvNextPerceptualLoss(
            device=self.device,
            model_type=ConvNextType.TINY,
            feature_layers=[0, 2, 4],  # Use fewer layers for faster testing
            use_gram=True,  # Test with style loss
            input_range=(-1, 1)  # Match Tanh output range
        )
        
        # Create optimizer
        self.optimizer = Adam(self.transform_net.parameters(), lr=0.001)

    def test_training_loop(self):
        """Test that the model can be trained and loss decreases"""
        batch_size = 2
        channels = 3
        height = 64  # Use smaller images for testing
        width = 64
        n_steps = 10
        
        # Generate random target images
        target_images = torch.randn(batch_size, channels, height, width,
                                  device=self.device)
        target_images = torch.tanh(target_images)  # Ensure target is in [-1, 1]
        
        # Keep track of losses
        losses = []
        
        # Training loop
        self.transform_net.train()
        for step in range(n_steps):
            # Generate input noise
            input_images = torch.randn(batch_size, channels, height, width,
                                     device=self.device)
            input_images = torch.tanh(input_images)  # Ensure input is in [-1, 1]
            
            # Forward pass
            transformed_images = self.transform_net(input_images)
            loss = self.perceptual_loss(transformed_images, target_images)
            losses.append(loss.item())
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print progress
            if step % 2 == 0:
                print(f"Step {step}, Loss: {loss.item():.6f}")
        
        # Test assertions
        self.assertGreater(len(losses), 1)
        self.assertLess(losses[-1], losses[0],
                       "Loss should decrease during training")
        print(f"Initial loss: {losses[0]:.6f}")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.2f}%")
        
        # Test that gradients are flowing
        for name, param in self.transform_net.named_parameters():
            self.assertIsNotNone(param.grad,
                               f"Gradient for {name} should not be None")
            grad_norm = torch.norm(param.grad).item()
            self.assertGreater(grad_norm, 0,
                             f"Gradient norm for {name} should be > 0")

    def test_dimensionality_preservation(self):
        """Test that the model preserves input dimensions exactly"""
        test_sizes = [
            (1, 3, 32, 32),
            (2, 3, 64, 64),
            (1, 3, 128, 128),
            (2, 3, 256, 256)
        ]
        
        self.transform_net.eval()
        for size in test_sizes:
            with self.subTest(size=size):
                input_images = torch.randn(*size, device=self.device)
                with torch.no_grad():
                    output_images = self.transform_net(input_images)
                
                # Test exact shape preservation
                self.assertEqual(
                    input_images.shape,
                    output_images.shape,
                    f"Output shape {output_images.shape} doesn't match input shape {input_images.shape}"
                )

    def test_model_output_range(self):
        """Test that the model outputs stay in the expected range"""
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        
        # Generate input noise
        input_images = torch.randn(batch_size, channels, height, width,
                                 device=self.device)
        input_images = torch.tanh(input_images)  # Ensure input is in [-1, 1]
        
        # Get model output
        self.transform_net.eval()
        with torch.no_grad():
            output_images = self.transform_net(input_images)
        
        # Test output range
        self.assertTrue(torch.all(output_images >= -1.0),
                       "Output should be >= -1")
        self.assertTrue(torch.all(output_images <= 1.0),
                       "Output should be <= 1")
        
        # Test output shape
        self.assertEqual(output_images.shape,
                        input_images.shape,
                        "Output shape should exactly match input shape")

    def test_gradient_flow(self):
        """Test gradient flow through both networks"""
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        
        # Generate input and target images
        input_images = torch.randn(batch_size, channels, height, width,
                                 device=self.device)
        input_images = torch.tanh(input_images)
        input_images.requires_grad_()
        
        target_images = torch.randn(batch_size, channels, height, width,
                                  device=self.device)
        target_images = torch.tanh(target_images)
        
        # Forward pass
        transformed_images = self.transform_net(input_images)
        loss = self.perceptual_loss(transformed_images, target_images)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        self.assertIsNotNone(input_images.grad,
                            "Input images should have gradients")
        
        for name, param in self.transform_net.named_parameters():
            self.assertIsNotNone(param.grad,
                               f"Parameter {name} should have gradients")
            
        # Verify shapes didn't change
        self.assertEqual(
            transformed_images.shape,
            input_images.shape,
            "Transformed image shape should exactly match input shape"
        )

if __name__ == '__main__':
    unittest.main()