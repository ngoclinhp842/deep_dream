'''
test_deep_dream.py

This script contains unit tests for the main.py module. It is designed to ensure
that the functions and logic within main.py behave as expected.

Author: Michelle Phan
Date: Fall 2024
Version: 1.0
'''

import unittest
import numpy as np
import tensorflow as tf
from PIL import Image
from deep_dream import DeepDream
from tf_util import load_pretrained_net, preprocess_image2tf

class TestDeepDream(unittest.TestCase):
    def test_image_loading(self):
        '''
        Test that the Miller image is loaded correctly.

        This test ensures that the Miller image is loaded and normalized
        correctly. The image should have a shape of (224, 224, 3) and
        pixel values should be between 0 and 1.
        '''
        # Load the Miller image
        img = Image.open('images/miller3_224x224.jpg')
        
        # Normalize the image
        generated_img = np.array(img) / 255
        
        # Check the shape of the image
        self.assertEqual(generated_img.shape, (224, 224, 3))
        
        # Check the range of the pixel values
        self.assertAlmostEqual(generated_img.min(), 0.0)
        self.assertAlmostEqual(generated_img.max(), 1.0)

    def test_vgg19_model_loading(self):
        '''
        Test VGG19 model loading.

        The test works as follows:
        1. Load the VGG19 model using the load_pretrained_net function.
        2. Verify that the model is not None.
        3. Get a list of the layer names in the model.
        4. Check that the length of the list of layer names is 22.
        5. Check that the first layer name is 'input_1'.
        6. Check that the last layer name is 'block5_pool'.
        '''
        pretrained_net = load_pretrained_net('vgg19')
        self.assertIsNotNone(pretrained_net)
        layer_names = [layer.name for layer in pretrained_net.layers]
        self.assertEqual(len(layer_names), 22)
        self.assertEqual(layer_names[0], 'input_1')
        self.assertEqual(layer_names[-1], 'block5_pool')

    def test_readout_model_creation(self):
        '''
        Test the creation of the readout model.

        This test verifies that the readout model is created correctly using the
        DeepDream class with a pretrained VGG19 network and selected layers containing
        'block5'. It ensures the model is not None and validates that the output of the
        model when given a random input has the expected number of activations and shape.

        The test works as follows:
        1. Create a random input with a shape of (1, 224, 224, 3).
        2. Create a DeepDream instance with the pretrained VGG19 network and selected
           layers containing 'block5'.
        3. Get the readout model from the DeepDream instance.
        4. Check that the readout model is not None.
        5. Pass the random input to the readout model and get its output.
        6. Check that the output has the expected number of activations and shape.
        '''
        # Create a random input
        rng = tf.random.Generator.from_seed(0)
        test_input = rng.uniform(shape=(1, 224, 224, 3))

        # Create the DeepDream instance and readout model
        pretrained_net = load_pretrained_net('vgg19')
        selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block5' in layer.name]
        readout_model = DeepDream(pretrained_net, selected_layer_names).readout_model

        # Check that the readout model is not None
        self.assertIsNotNone(readout_model)

        # Get the output of the readout model given the random input
        rng = tf.random.Generator.from_seed(0)
        test_input = rng.uniform(shape=(1, 224, 224, 3))
        test_net_acts = readout_model(test_input)

        # Check the number of activations and shape of the output
        self.assertEqual(len(test_net_acts), 5)
        self.assertEqual(test_net_acts[0].shape, (1, 14, 14, 512))

    def test_deepdream_network_creation(self):
        '''
        Test the creation of the DeepDream instance.

        This test works as follows:
        1. Load the VGG19 model using the load_pretrained_net function.
        2. Get a list of the layer names in the model that contain 'block5'.
        3. Create a DeepDream instance with the VGG19 model and selected layer names.
        4. Check that the DeepDream instance is not None.
        '''
        pretrained_net = load_pretrained_net('vgg19')
        selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block5' in layer.name]
        test_dd = DeepDream(pretrained_net, selected_layer_names)
        self.assertIsNotNone(test_dd)

    def test_deepdream_network_fitting(self):
        '''
        Test the fitting of the DeepDream network.

        This test verifies that the DeepDream network can be fit to a given image
        for one epoch with a learning rate of 0.1. It ensures that the loss history
        has length 1.

        The test works as follows:
        1. Load the VGG19 model and select the layers that contain 'block5'.
        2. Create a DeepDream instance with the VGG19 model and selected layer names.
        3. Load the Hokusai image and preprocess it.
        4. Create a tf.Variable with the preprocessed image.
        5. Fit the DeepDream network to the image for one epoch with a learning rate
           of 0.1.
        6. Ensure the loss history has length 1.
        '''
        # Load the VGG19 model and select the layers that contain 'block5'
        pretrained_net = load_pretrained_net('vgg19')
        selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block5' in layer.name]

        # Create a DeepDream instance with the VGG19 model and selected layer names
        test_dd = DeepDream(pretrained_net, selected_layer_names)

        # Load the Hokusai image and preprocess it
        img = Image.open('images/hokusai.jpg')
        img = img.resize((224, 224))
        img = np.array(img) / 255
        generated_img = preprocess_image2tf(img, as_var=True)

        # Create a tf.Variable with the preprocessed image
        gen_img_1epoch = tf.Variable(tf.identity(generated_img))

        # Fit the DeepDream network to the image for one epoch with a learning rate of 0.1
        loss_hist = test_dd.fit(gen_img_1epoch, n_epochs=1, lr=0.1)

        # Ensure the loss history has length 1
        self.assertEqual(len(loss_hist), 1)

if __name__ == '__main__':
    unittest.main()