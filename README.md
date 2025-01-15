# DEEP DREAM 

Use transfer learning with pretrained neural network **VGG19** (low-level TensorFlow API) to implement the DeepDream algorithm, which modifies images based on neural activation patterns.

## 🚀 Running DeepDream:
To run DeepDream on the file main.py 

```sh
python3 main.py
```

## 👀 Example Running deep_dream_extension.py:
- Original Image
<p align="center">
  <img align="center" alt="Original Image" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/images/hokusai.jpg">
</p>

- Output of generating image when only early layer block1 is involved in DeepDream process
<p align="center">
  <img align="center" alt="Output of generating image when only layer block1 is involved in DeepDream process" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/outputs/transfer_learning_layer1.png">
</p>

- Output of generating image when only later layer block4 is involved in DeepDream process
<p align="center">
  <img align="center" alt="Output of generating image when only layer block4 is involved in DeepDream process" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/outputs/transfer_learning_layer_block4.png">
</p>

- Output of generating image when only pooling layer is involved in DeepDream process
<p align="center">
  <img align="center" alt="Output of generating image when only pooling is involved in DeepDream process" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/outputs/transfer_learning_pooling_layers.png">
</p>

## 📝 Analysis:

1. One key difference in the generated images when only the earlier/later network layers are involved in the Deep Dream process

Earlier layer capture low-level features like edges, textures, and simple patterns. This results in highly repetitive and grid-like patterns, as seen in the dots and fine details.
These layers operate on small receptive fields and are focused on localized patterns, leading to intricate but less meaningful shapes.

<p align="center">
  <img align="center" alt="Output of generating image when only layer block1 is involved in DeepDream process" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/outputs/transfer_learning_layer1.png">
</p>

Later latyer capture high-level features such as object shapes, structures, and semantic patterns.
You can observe more complex and meaningful shapes, such as animal-like forms, which reflect the higher-level abstractions learned by these layers.

<p align="center">
  <img align="center" alt="Output of generating image when only layer block4 is involved in DeepDream process" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/outputs/transfer_learning_layer_block4.png">
</p>

2. One key difference in the generated images when you let pooling vs different conv layers contribute to the generated image

Max pooling layers summarize features by selecting the most activated elements across spatial areas, leading to more repetitive, grid-like patterns.
Pooling layers reduce spatial resolution, causing finer spatial details to be discarded.
This results in a more generalized visualization, as you can see the mountain is disappearing gradually as we trained more npoches.

<p align="center">
  <img align="center" alt="Output of generating image when only pooling is involved in DeepDream process" width="400" src="https://github.com/ngoclinhp842/deep_dream/blob/main/deep_dream/outputs/transfer_learning_pooling_layers.png">
</p>

## ⚖️ License:
Apache License 2.0
