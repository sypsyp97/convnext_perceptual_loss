import unittest
import torch
from convnext_perceptual_loss import ConvNextPerceptualLoss, ConvNextType, make_dummy_data

class TestConvNextPerceptualLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up common test fixtures"""
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cls.batch_size = 2
        cls.height = 256
        cls.width = 256
        
        # Create loss function instances with different configurations
        cls.loss_fn_default = ConvNextPerceptualLoss(
            device=cls.device,
            model_type=ConvNextType.TINY,
            use_gram=False,
            feature_layers=[0, 2]  # Using fewer layers for faster testing
        )
        
        # Create Gram matrix loss with same layers but different weights
        cls.loss_fn_gram = ConvNextPerceptualLoss(
            device=cls.device,
            model_type=ConvNextType.TINY,
            use_gram=True,
            feature_layers=[0, 2],
            feature_weights=[0.7, 0.3]  # Different weights to ensure different behavior
        )
        
        cls.loss_fn_custom = ConvNextPerceptualLoss(
            device=cls.device,
            model_type=ConvNextType.TINY,
            feature_layers=[0, 2, 4],
            feature_weights=[0.5, 0.3, 0.2],
            input_range=(0, 1)
        )

    def test_input_shapes(self):
        """Test handling of different input shapes"""
        test_shapes = [
            (self.batch_size, 3, self.height, self.width),  # RGB
            (self.batch_size, 1, self.height, self.width),  # Grayscale
            (1, 3, self.height, self.width),               # Single RGB image
            (1, 1, self.height, self.width)                # Single grayscale image
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                input_tensor = make_dummy_data(*shape, device=self.device)
                target_tensor = make_dummy_data(*shape, device=self.device)
                
                loss = self.loss_fn_default(input_tensor, target_tensor)
                
                self.assertIsInstance(loss, torch.Tensor)
                self.assertEqual(loss.dim(), 0)  # Should be a scalar
                self.assertGreaterEqual(loss.item(), 0)  # Loss should be non-negative

    def test_input_ranges(self):
        """Test handling of different input value ranges"""
        # Test with [-1, 1] range (default)
        input_neg1_1 = torch.rand(self.batch_size, 3, self.height, self.width, 
                                device=self.device) * 2 - 1
        target_neg1_1 = torch.rand(self.batch_size, 3, self.height, self.width, 
                                 device=self.device) * 2 - 1
        loss_neg1_1 = self.loss_fn_default(input_neg1_1, target_neg1_1)
        
        # Test with [0, 1] range
        input_0_1 = torch.rand(self.batch_size, 3, self.height, self.width, 
                             device=self.device)
        target_0_1 = torch.rand(self.batch_size, 3, self.height, self.width, 
                              device=self.device)
        loss_0_1 = self.loss_fn_custom(input_0_1, target_0_1)
        
        self.assertIsInstance(loss_neg1_1, torch.Tensor)
        self.assertIsInstance(loss_0_1, torch.Tensor)

    def test_gram_matrix(self):
        """Test Gram matrix computation"""
        # Create structured input with clear patterns
        input_tensor = torch.zeros(self.batch_size, 3, self.height, self.width, 
                                 device=self.device)
        # Add structured pattern to input
        input_tensor[:, 0, :self.height//2] = 1.0  # Red channel pattern
        input_tensor[:, 1, self.height//2:] = 1.0  # Green channel pattern
        
        # Create differently structured target
        target_tensor = torch.zeros(self.batch_size, 3, self.height, self.width, 
                                  device=self.device)
        # Add different pattern to target
        target_tensor[:, 0, :, :self.width//2] = 1.0  # Vertical pattern
        target_tensor[:, 1, :, self.width//2:] = 1.0  # Different vertical pattern
        
        # Compute both types of losses
        loss_with_gram = self.loss_fn_gram(input_tensor, target_tensor)
        loss_without_gram = self.loss_fn_default(input_tensor, target_tensor)
        
        # Verify that both losses give valid results
        self.assertIsInstance(loss_with_gram, torch.Tensor)
        self.assertIsInstance(loss_without_gram, torch.Tensor)
        self.assertGreater(loss_with_gram.item(), 0)
        self.assertGreater(loss_without_gram.item(), 0)
        
        # Compute the relative difference between losses
        relative_diff = abs(loss_with_gram.item() - loss_without_gram.item()) / max(loss_with_gram.item(), loss_without_gram.item())
        # Assert that the relative difference is significant (>10%)
        self.assertGreater(relative_diff, 0.1, "Gram matrix loss should be significantly different from direct feature loss")

    def test_custom_layers_weights(self):
        """Test custom layer selection and weights"""
        input_tensor = make_dummy_data(self.batch_size, 3, self.height, self.width, 
                                     device=self.device)
        target_tensor = make_dummy_data(self.batch_size, 3, self.height, self.width, 
                                      device=self.device)
        
        # Compare default and custom layer configurations
        loss_default = self.loss_fn_default(input_tensor, target_tensor)
        loss_custom = self.loss_fn_custom(input_tensor, target_tensor)
        
        self.assertIsInstance(loss_default, torch.Tensor)
        self.assertIsInstance(loss_custom, torch.Tensor)
        self.assertNotEqual(loss_default.item(), loss_custom.item())

    def test_make_dummy_data(self):
        """Test dummy data generation function"""
        test_params = [
            {'batch_size': 1, 'channels': 3, 'height': 256, 'width': 256},
            {'batch_size': 2, 'channels': 1, 'height': 128, 'width': 128},
            {'batch_size': 4, 'channels': 3, 'height': 64, 'width': 64}
        ]
        
        for params in test_params:
            with self.subTest(params=params):
                dummy_data = make_dummy_data(**params, device=self.device)
                
                self.assertIsInstance(dummy_data, torch.Tensor)
                self.assertEqual(dummy_data.shape, 
                               (params['batch_size'], params['channels'], 
                                params['height'], params['width']))
                # Check device type only, not index
                self.assertEqual(dummy_data.device.type, self.device.type)

    def test_identical_inputs(self):
        """Test loss computation with identical inputs"""
        input_tensor = make_dummy_data(self.batch_size, 3, self.height, self.width, 
                                     device=self.device)
        
        loss = self.loss_fn_default(input_tensor, input_tensor)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertLess(loss.item(), 1e-5, "Loss should be very close to zero for identical inputs")

    def test_model_types(self):
        """Test initialization with different model types"""
        for model_type in ConvNextType:
            with self.subTest(model_type=model_type):
                loss_fn = ConvNextPerceptualLoss(
                    device=self.device,
                    model_type=model_type
                )
                
                input_tensor = make_dummy_data(1, 3, self.height, self.width, 
                                             device=self.device)
                target_tensor = make_dummy_data(1, 3, self.height, self.width, 
                                              device=self.device)
                
                loss = loss_fn(input_tensor, target_tensor)
                
                self.assertIsInstance(loss, torch.Tensor)
                self.assertGreaterEqual(loss.item(), 0)

if __name__ == '__main__':
    unittest.main()