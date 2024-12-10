'''deep_dream.py
Generate art with a pretrained neural network using the DeepDream algorithm
Michelle Phan and Varsha Yarram
CS 343: Neural Networks
Project 4: Transfer Learning
Fall 2024
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import tf_util


class DeepDream:
    '''Runs the DeepDream algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    '''
    def __init__(self, pretrained_net, selected_layers_names):
        '''DeepDream constructor.

        Parameters:
        -----------
        pretrained_net: TensorFlow Keras Model object. Pretrained network configured to return netAct values in
            ALL layers when presented with an input image.
        selected_layers_names: Python list of str. Names of layers in `pretrained_net` that we will readout netAct values
            from in order to contribute to the generated image.

        TODO:
        1. Define instance variables for the pretrained network and the number of selected layers used to readout netAct.
        2. Make an readout model for the selected layers (use function in `tf_util`) and assign it as an instance variable.
        '''
        self.loss_history = None

        self.pretrained_net = pretrained_net
        self.selected_layers_names = selected_layers_names
        self.readout_model = tf_util.make_readout_model(self.pretrained_net, self.selected_layers_names)

    def loss_layer(self, layer_net_acts):
        '''Computes the contribution to the total loss from the current layer with netAct values `layer_net_acts`.

        The loss contribution is the mean of all the netAct values in the current layer.

        Parameters:
        -----------
        layer_net_acts: tf tensor. shape=(1, Iy, Ix, K). The netAct values in the current selected layer. K is the
            number of kernels in the layer.

        Returns:
        -----------
        loss component from current layer. float. Mean of all the netAct values in the current layer.
        '''
        # Compute the mean of all netAct values
        loss = tf.reduce_mean(layer_net_acts)

        # Return as a Python float
        return loss
        

    def forward(self, gen_img, standardize_grads=True, eps=1e-8):
        '''Performs forward pass through the pretrained network with the generated image `gen_img`.
        Loss is computed based on the SELECTED layers (in readout model).

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. The singleton dimension is the batch dimension (N).
        standardize_grads: bool. Should we standardize the image gradients?
        eps: float. Small number used in standardization to prevent possible division by 0 (i.e. if stdev is 0).

        Returns:
        -----------
        loss. float. Sum of the loss components from all the selected layers.
        grads. shape=(1, Iy, Ix, n_chans). Image gradients (`dImage` aka `dloss_dImage`) — gradient of the
            generated image with respect to each of the pixels in the generated image.

        TODO:
        While tracking gradients:
        - Use the readout model to extract the netAct values in the selected layers for `gen_img`.
        - Compute the average loss across all selected layers.
        Then:
        - Obtain the tracked gradients of the loss with respect to the generated image.
        '''
        with tf.GradientTape() as tape:
            # Track the generated image for gradient computation
            tape.watch(gen_img)

            # Use the readout model to extract netAct values from the selected layers
            net_act_values = self.readout_model(gen_img)

            # Compute the average loss across all selected layers
            losses = [self.loss_layer(act) for act in net_act_values]
            loss = tf.math.reduce_mean(losses)

        # Compute the gradients of the loss with respect to the generated image
        grads = tape.gradient(loss, gen_img)

        if standardize_grads:
            # Standardize gradients to have mean 0 and standard deviation 1
            grad_mean = tf.reduce_mean(grads)
            grad_std = tf.math.reduce_std(grads)
            grads = (grads - grad_mean) / (grad_std + eps)

        return loss, grads

    def fit(self, gen_img, n_epochs=26, lr=0.01, print_every=25, plot=True, plot_fig_sz=(5, 5), export=True):
        '''Iteratively modify the generated image (`gen_img`) for `n_epochs` with the image gradients using the
            gradient ASCENT algorithm. In other words, run DeepDream on the generated image.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current epoch) every this many epochs.
        plot: bool. If true, plot/show the generated image `print_every` epochs.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` epochs. Each exported image should have the current epoch number in the filename so that
            the image currently exported image doesn't overwrite the previous one. For example, image_1.jpg, image_2.jpg,
            etc.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Compute the forward pass on the generated image for `n_epochs`.
        2. Apply the gradients to the generated image using the gradient ASCENT update rule.
        3. Clip pixel values to the range [0, 1] and update the generated image.
            The TensorFlow `assign` function is helpful here because = would "wipe out" the tf.Variable property,
            which is not what we want because we want to track gradients on the generated image across epochs.
        4. After the first epoch completes, always print out how long it took to finish the first epoch and an estimate
        of how long it will take to complete all the epochs (in minutes).

        NOTE:
        - Deep Dream performs gradient ASCENT rather than DESCENT (which we are more used to). The difference is only
        in the sign of the gradients.
        - Clipping is different than normalization!
        '''
        self.loss_history = []
        start_time = time.time()

        for epoch in range(1, n_epochs + 1):
            # Compute forward pass and get loss and gradients
            loss, grads = self.forward(gen_img, standardize_grads=True)
            self.loss_history.append(loss.numpy())

            # Apply gradient ascent update rule
            gen_img.assign_add(lr * grads)

            # Clip the pixel values to the range [0, 1]
            gen_img.assign(tf.clip_by_value(gen_img, 0.0, 1.0))

            # Print progress
            if epoch == 1:
                elapsed = time.time() - start_time
                estimated_total_time = (elapsed / epoch) * n_epochs / 60
                print(f"Epoch {epoch}/{n_epochs}: Loss = {loss.numpy():.5f} (Estimated total time: {estimated_total_time:.2f} minutes)")
            elif epoch % print_every == 0:
                print(f"Epoch {epoch}/{n_epochs}: Loss = {loss.numpy():.5f}")

            # Plot the generated image
            if plot and epoch % print_every == 0:
                plt.figure(figsize=plot_fig_sz)
                plt.imshow(gen_img[0])
                plt.axis('off')
                plt.show()

            # Export the image
            if export and epoch % print_every == 0:
                output_dir = "deep_dream_output"
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"image_{epoch}.jpg")
                tf.keras.preprocessing.image.save_img(filename, gen_img[0].numpy())

        return self.loss_history


    def fit_multiscale(self, gen_img, n_scales=4, scale_factor=1.3, n_epochs=26, lr=0.01, print_every=1, plot=True,
                       plot_fig_sz=(5, 5), export=True):
        '''Run DeepDream `fit` on the generated image `gen_img` a total of `n_scales` times. After each time, scale the
        width and height of the generated image by a factor of `scale_factor` (round to nearest whole number of pixels).
        The generated image does NOT start out from scratch / the original image after each resizing. Any modifications
        DO carry over across runs.

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients.
        n_scales: int. Number of times the generated image should be resized and DeepDream should be run.
        n_epochs: int. Number of epochs to run gradient ascent on the generated image.
        lr: float. Learning rate.
        print_every: int. Print out progress (current scale) every this many SCALES (not epochs).
        plot: bool. If true, plot/show the generated image `print_every` SCALES.
        plot_fig_sz: tuple of ints. The plot figure size (height, width) to use when plotting/showing the generated image.
        export: bool. Whether to export a JPG image to the `deep_dream_output` folder in the working directory
            every `print_every` SCALES. Each exported image should have the current scale number in the filename so that
            the image currently exported image doesn't overwrite the previous one.

        Returns:
        -----------
        self.loss_history. Python list of float. Loss values computed on every epoch of training.

        TODO:
        1. Call fit `n_scale` times. Pass along hyperparameters (n_epochs, etc.). Turn OFF plotting and exporting within
        the `fit` method — this method should take over the plotting and exporting (in scale intervals rather than epochs).
        2. Multiplicatively scale the generated image.
            Check out: https://www.tensorflow.org/api_docs/python/tf/image/resize

            NOTE: The output of the built-in resizing process is NOT a tf.Variable (its an ordinary tensor).
            But we need a tf.Variable to compute the image gradient during gradient ascent.
            So, wrap the resized image in a tf.Variable.
        3. After the first scale completes, always print out how long it took to finish the first scale and an estimate
        of how long it will take to complete all the scales (in minutes).
        ''' 
        self.loss_history = []
        start_time = time.time()

        for scale in range(1, n_scales + 1):
            # Run the fit method for the current scale
            self.fit(gen_img, n_epochs=n_epochs, lr=lr, print_every=n_epochs, plot=False, export=False)

            # Print progress
            if scale == 1:
                elapsed = time.time() - start_time
                estimated_total_time = (elapsed / scale) * n_scales / 60
                print(f"Scale {scale}/{n_scales}: Completed (Estimated total time: {estimated_total_time:.2f} minutes)")
            elif scale % print_every == 0:
                print(f"Scale {scale}/{n_scales}: Completed")

            # Plot the generated image
            if plot:
                plt.figure(figsize=plot_fig_sz)
                plt.imshow(gen_img[0])
                plt.axis('off')
                plt.show()

            # Export the image
            if export and scale % print_every == 0:
                output_dir = "deep_dream_output"
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"scale_{scale}.jpg")
                tf.keras.preprocessing.image.save_img(filename, gen_img[0].numpy())

            # Resize the image for the next scale
            new_size = [int(gen_img.shape[1] * scale_factor), int(gen_img.shape[2] * scale_factor)]
            resized_img = tf.image.resize(gen_img, new_size)
            gen_img = tf.Variable(resized_img, dtype=tf.float32)


        return self.loss_history
        
