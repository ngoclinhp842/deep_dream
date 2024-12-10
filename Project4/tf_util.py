'''tf_util.py
Helper/utility functions related to using TensorFlow for Transfer Learning and working with images
Michelle Phan and Varsha Yarram
CS 343: Neural Networks
Project 4: Transfer Learning
Fall 2024
'''
import numpy as np
from PIL import Image
import tensorflow as tf


def load_pretrained_net(net_name='vgg19'):
    '''Loads the pretrained network (included in Keras) identified by the string `net_name`.

    Parameters:
    -----------
    net_name: str. Name of pretrained network to load. By default, this is VGG19.

    Returns:
    -----------
    The pretained net. Keras object.

    NOTE: Pretrained net should NOT be trainable and NOT include the output layer.
    '''
    nets = {
        "vgg19": tf.keras.applications.VGG19,
    }

    net_name = net_name.lower()
    if net_name not in nets:
        raise ValueError(f"Unsupported network name '{net_name}'. Supported names: {list(nets.keys())}")
    
    pretrained_net = nets[net_name](include_top=False)
    pretrained_net.trainable = False
    return pretrained_net


def get_all_layer_strs(pretrained_net):
    '''Gets the complete list of layer names from the pretrained net.

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network.

    Returns:
    -----------
    Python list of str. Length is the number of layers in the pretrained network.
    '''
    return [layer.name for layer in pretrained_net.layers]

    

def filter_layer_strs(layer_names, match_str='conv4'):
    '''Extracts the layer name strs from `layer_names` (the complete list) that have `match_str` in the name.

    Parameters:
    -----------
    layer_names: Python list of str. The complete list of layer names in the pretrained network
    match_str: str. Substring searched for within each layer name

    Returns:
    -----------
    Python list of str. The list of layers from `layer_names` that include the string `match_str`
    '''
    # Filter and return layer names that include `match_str`
    return [name for name in layer_names if match_str in name]


def preprocess_image2tf(img, as_var):
    '''Converts an image (in numpy ndarray format) to TensorFlow tensor format.

    Parameters:
    -----------
    img: ndarray. shape=(Iy, Ix, n_chans). A single image.
    as_var: bool. Indicates if the tensor should be represented as a tf.Variable.

    Returns:
    -----------
    tf tensor. dtype=tf.float32. shape=(1, Iy, Ix, n_chans).

    NOTE: Adds a leading singleton batch dimension to the returned tf tensor.
    '''
    # Convert the image to tf.float32 and add a batch dimension
    tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)  # Add singleton batch dimension
    
    # Convert to tf.Variable if as_var is True
    if as_var:
        tensor = tf.Variable(tensor)

    return tensor


def make_readout_model(pretrained_net, layer_names):
    '''Makes a tf.keras.Model object that returns the netAct (output) values of layers in the pretrained model
    `pretrained_net` that have the names in the list `layer_names` (the readout model).

    Parameters:
    -----------
    pretrained_net: Keras object. The pretrained network
    layer_names: Python list of str. Selected list of pretrained net layer names whose netAct values should be returned
        by the readout model.

    Returns:
    -----------
    tf.keras.Model object (readout model) that provides a readout of the netAct values in the selected layer list
        (`layer_names`).
    '''
    # Get the outputs of the specified layers
    outputs = [pretrained_net.get_layer(name).output for name in layer_names]

    # Create a new model with the same input as the pretrained network
    readout_model = tf.keras.Model(inputs=pretrained_net.input, outputs=outputs)

    return readout_model


def tf2image(tensor):
    '''Converts a TensorFlow tensor into a PIL Image object.

    Parameters:
    -----------
    tensor: tf tensor. dtype=tf.float32. shape=(1, Iy, Ix, n_chans). A single image. Values range from 0-1.

    Returns:
    -----------
    PIL Image object. dtype=uint8. shape=(Iy, Ix, n_chans). Image representation of the input tensor with pixel values
        between 0 and 255 (unsigned ints).

    NOTE:
    - Scale pixel values to the range [0, 255] BEFORE converting to uint8 dtype.
    - Remove batch (singleton) dimension (if present)
    - One way to convert to PIL Image is to first convert to numpy ndarray.

    The following should be helpful:
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray
    '''
    pass
