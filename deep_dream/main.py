'''
main.py

This script serves as the entry point for the program. It contains the main() function,
which is executed when the script is run directly. Additional logic and functionality
can be added within the main() function or through imported modules.

Author: Michelle Phan
Date: Fall 2024
Version: 1.0
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import tf_util
from deep_dream import DeepDream

plt.show()
plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 16})

np.set_printoptions(suppress=True, precision=3)

def plot_img(img, title=''):
    """
    Plot a given image with a given title.

    Parameters:
    -----------
    img: ndarray. shape=(Iy, Ix, n_chans). The image to be plotted.
    title: str. The title of the plot.
    """
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])  # Remove x-ticks
    plt.yticks([])  # Remove y-ticks
    plt.show()

def main():
    '''
    The main function of the program.

    This function serves as the starting point for the application. Add your main logic
    or function calls within this function.
    '''
    # Output1: run DeepDream on Hokusai test image at 224x224
    # load in and preprocess Hokusai test image at 224x224
    img = Image.open('images/hokusai.jpg')
    # Resize the image
    img = img.resize((224, 224))

    img = np.array(img) / 255
    plot_img(img, 'Hokusai Original Image')
    generated_img = tf_util.preprocess_image2tf(img, as_var=True)
    gen_img_1epoch = tf.Variable(tf.identity(generated_img))

    # load in pretrained VGG19 network
    pretrained_net = tf_util.load_pretrained_net('vgg19')

    # get lists of layers with 'block4' in their name
    selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block4' in layer.name]

    # create DeepDream network, only activate layers with 'block5' in their name
    test_dd = DeepDream(pretrained_net, selected_layer_names)
    # fit DeepDream network to the image over 10 epochs and with a learning rate of 0.1
    loss_hist = test_dd.fit(gen_img_1epoch, n_epochs=5, lr=0.1, print_every=5)


    # Output2: load in and preprocess Hokusai test image at 224x224
    img = Image.open('images/hokusai.jpg')
    # Resize the image
    img = img.resize((224, 224))

    img = np.array(img) / 255
    plot_img(img, 'Hokusai Original Image')
    generated_img = tf_util.preprocess_image2tf(img, as_var=True)
    gen_img_1epoch = tf.Variable(tf.identity(generated_img))

    # load in pretrained VGG19 network
    pretrained_net = tf_util.load_pretrained_net('vgg19')

    # get lists of layers with 'block4' in their name
    selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block4' in layer.name]

    # create DeepDream network, only activate layers with 'block5' in their name
    test_dd = DeepDream(pretrained_net, selected_layer_names)

    # fit DeepDream network to the image over 10 epochs and with a learning rate of 0.1
    loss_hist = test_dd.fit_multiscale(gen_img_1epoch, n_epochs=5, lr=0.1, print_every=5, plot=True)

    # Outpu3: load in and preprocess Hokusai test image at 224x224

    img = np.array(img) / 255
    generated_img = tf_util.preprocess_image2tf(img, as_var=True)
    gen_img_1epoch = tf.Variable(tf.identity(generated_img))

    # load in pretrained VGG19 network
    pretrained_net = tf_util.load_pretrained_net('vgg19')

    # get lists of layers with 'block4' in their name
    selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block1' in layer.name]

    # create DeepDream network, only activate layers with 'block5' in their name
    test_dd = DeepDream(pretrained_net, selected_layer_names)
    # fit DeepDream network to the image over 10 epochs and with a learning rate of 0.1
    print('Training with early layer')
    loss_hist = test_dd.fit_multiscale(gen_img_1epoch, n_epochs=5, lr=0.1, print_every=5, plot=True)

    print('Training with late layer')
    gen_img_1epoch = tf.Variable(tf.identity(generated_img))
    generated_img = tf_util.preprocess_image2tf(img, as_var=True)
    selected_layer_names = [layer.name for layer in pretrained_net.layers if 'block5' in layer.name]
    test_dd = DeepDream(pretrained_net, selected_layer_names)
    loss_hist = test_dd.fit_multiscale(gen_img_1epoch, n_epochs=5, lr=0.1, print_every=5, plot=True)



if __name__ == "__main__":
    '''
    Entry point of the script.

    This condition ensures that the main() function is only executed when the script is
    run directly (not imported as a module).
    '''
    main()