'''deep_dream.py
Generate art with a pretrained neural network using the DeepDream algorithm
Extension: 
1. Use CNN filters instead of layers to 'dream' the generated image 
2. Visulize CNN filters to allow user to select which features/patterns to 'dream'
Michelle Phan and Varsha Yarram
CS 343: Neural Networks
Project 4: Transfer Learning
Fall 2024
'''
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

import tf_util


class DeepDreamExtension:
    '''Runs the DeepDream algorithm on an image using a pretrained network.
    You should NOT need to import and use Numpy in this file (use TensorFlow instead).
    DeepDreamExtension is an extension of DeepDream class in deep_dream.py, allow user to select whether
    to use CNN filters instead of layers to allow user to select which features/patterns to 'dream'
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
        print(self.readout_model.input)

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
        
    def loss_filter(self, layer_net_acts, filter_indices):
        """Computes the mean activation for specific filters in a layer.

        Parameters:
        -----------
        layer_net_acts: tf.Tensor
            Activations of a specific layer from the model, shape=(batch, height, width, channels).
        filter_indices: list of int
            Indices of the filters to compute the mean activation for.

        Returns:
        -----------
        float
            Mean activation of the selected filters.
        """
        selected_filters = tf.gather(layer_net_acts, filter_indices, axis=-1)
        return tf.reduce_mean(selected_filters)
    
    def visualize_filters(self, layer_names, filter_indices, n_epochs=40, lr=0.01, img_size=(224, 224)):
        """Visualizes the features that specific CNN filters learn at any depth of the network.

        Parameters:
        -----------
        layer_names: list of str
            Names of the layers in the pretrained network to visualize filters.
        filter_indices: list of int
            Indices of the filters to visualize within the selected layer.
        n_epochs: int
            Number of epochs to run gradient ascent.
        lr: float
            Learning rate for gradient ascent.
        img_size: tuple of int
            Size of the input image for visualization (height, width).

        Returns:
        -----------
        None
        """
        # Determine the input shape
        input_shape = self.readout_model.input.shape[1:]
        noise_img = tf.Variable(np.random.uniform(0, 1, size=(1, *input_shape)).astype(np.float32))

        # Initialize subplots
        num_layers = len(layer_names)
        num_filters = len(filter_indices)
        num_cols = min(4, num_filters)  # Max 4 columns
        num_rows = num_layers * ((num_filters + num_cols - 1) // num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_layers * 5))
        axes = axes.flatten()

        subplot_idx = 0

        for layer_name in layer_names:
            # Create a sub-model targeting the specified layer
            target_layer = self.readout_model.get_layer(layer_name).output
            sub_model = tf.keras.Model(inputs=self.readout_model.input, outputs=target_layer)

            # Pass the random noise image through the sub-model
            feature_maps = sub_model(noise_img)

            # Visualize the selected filters
            for filter_idx in filter_indices:
                ax = axes[subplot_idx]
                ax.imshow(feature_maps[0, :, :, filter_idx], cmap='viridis')
                ax.axis('off')
                ax.set_title(f'{layer_name} - Filter {filter_idx}')
                subplot_idx += 1

        # Hide any unused subplots
        for idx in range(subplot_idx, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()


    def forward(self, gen_img, standardize_grads=True, eps=1e-8, target_layers=None):
        '''Performs forward pass through the pretrained network with the generated image `gen_img`.
        Loss is computed based on the SELECTED layers (in readout model).

        Parameters:
        -----------
        gen_img: tf tensor. shape=(1, Iy, Ix, n_chans). Generated image that is used to compute netAct values, loss,
            and the image gradients. The singleton dimension is the batch dimension (N).
        standardize_grads: bool. Should we standardize the image gradients?
        eps: float. Small number used in standardization to prevent possible division by 0 (i.e. if stdev is 0).
        target_layers: dict, optional
            Dictionary mapping layer names to lists of filter indices. Specifies which layers and filters
            to use for influencing the generated image. If None, all layers are used.

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

            loss = 0

            if target_layers is not None:
                for layer_name, filter_indices in target_layers.items():
                    # Extract activations from the specific layer
                    target_layer = self.readout_model.get_layer(layer_name).output
                    sub_model = tf.keras.Model(inputs=self.readout_model.input, outputs=target_layer)
                    net_act_values = sub_model(gen_img)

                    if filter_indices is not None:
                        # Compute loss for specific filters
                        loss += tf.reduce_mean(tf.gather(net_act_values, filter_indices, axis=-1))
                    else:
                        # Compute loss for all filters in the layer
                        loss += tf.reduce_mean(net_act_values)
            else:
                # Use all layers in the readout model
                net_act_values = self.readout_model(gen_img)
                loss += tf.reduce_sum([tf.reduce_mean(act) for act in net_act_values])

        # Compute the gradients of the loss with respect to the generated image
        grads = tape.gradient(loss, gen_img)

        if standardize_grads:
            # Standardize gradients to have mean 0 and standard deviation 1
            grad_mean = tf.reduce_mean(grads)
            grad_std = tf.math.reduce_std(grads)
            grads = (grads - grad_mean) / (grad_std + eps)

        return loss, grads

    def fit(self, gen_img, n_epochs=26, lr=0.01, print_every=25, plot=True, plot_fig_sz=(5, 5), export=True, target_layers=None):
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
        target_layers: dict, optional
            Dictionary mapping layer names to lists of filter indices. Specifies which layers and filters
            to use for influencing the generated image. If None, all layers are used.

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
            if target_layers is not None:
                loss, grads = self.forward(gen_img, standardize_grads=True, target_layers=target_layers)
            else:
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
                       plot_fig_sz=(5, 5), export=True, target_layers=None):
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
        target_layers: dict, optional
            Dictionary mapping layer names to lists of filter indices. Specifies which layers and filters
            to use for influencing the generated image. If None, all layers are used.

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
            self.fit(gen_img, n_epochs=n_epochs, lr=lr, print_every=n_epochs, plot=False, export=False, target_layers=target_layers)

            # Print progress
            if scale == 1:
                elapsed = time.time() - start_time
                estimated_total_time = (elapsed / scale) * n_scales / 60
                print(f"Scale {scale}/{n_scales}: Completed (Estimated total time: {estimated_total_time:.2f} minutes)")
            elif scale % print_every == 0:
                print(f"Scale {scale}/{n_scales}: Completed")

            # Plot the generated image
            if plot and scale % print_every == 0:
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
        
