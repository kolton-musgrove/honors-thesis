# Cascade Neural Network Inference

## Overview

This repository contains a Jupyter notebook demonstrating a cascade neural network inference system for image classification. The system uses three progressively more complex neural networks (MLP, LeNet-5, and AlexNet) arranged in a cascade configuration to optimize the trade-off between classification accuracy and computational efficiency.

## Key Features

- Implementation of three neural network architectures:
  - MLP (Multi-Layer Perceptron): A simple feedforward neural network
  - LeNet-5: A classic convolutional neural network
  - AlexNet: A deeper, more complex CNN architecture
- Cascade inference system that:
  - Starts with the simplest model (MLP)
  - Only processes inputs with more complex models if necessary
  - Uses confidence thresholds to determine whether to pass to the next model
- Comprehensive analysis of the system performance:
  - Per-model inference times
  - Pass-through rates between models
  - Overall system efficiency metrics

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- tqdm

## How It Works

1. The system first trains all three neural networks on the CIFAR-10 dataset
2. For each input image:
   - First, it's processed by the simplest model (MLP)
   - If the MLP's confidence is below a threshold, the input is passed to LeNet-5
   - If LeNet-5's confidence is still below the threshold, it's passed to AlexNet
3. Performance metrics are collected and analyzed, including:
   - Inference time for each model
   - Number of samples handled by each model
   - Overall system efficiency compared to always using the most complex model

## Results

The notebook includes detailed visualizations and statistics that show:

- Distribution of inference times for each model
- Percentage of samples handled by each model in the cascade
- Average inference time per sample in the cascade system

The results demonstrate that while this particular configuration of the cascade system doesn't provide a speed improvement over always using the simplest model, it illustrates the methodology for building cascading inference systems that could be optimized for different accuracy/speed trade-offs.

```plaintext
---- Cascade Efficiency Analysis ----
Total samples: 10000
MLP handled exclusively: 891 samples (8.91%)
Passed to LeNet-5: 9109 samples (91.09%)
Passed to AlexNet: 5538 samples (55.38%)

Average cascade inference time per sample: 57.73 ms
If all samples used only MLP: 23.66 ms per sample
If all samples used only AlexNet: 24.28 ms per sample
Speedup from using cascade: 0.42x
```

## Limitations

The current implementation actually shows a slowdown compared to using a single model, which highlights the importance of carefully selecting confidence thresholds and ensuring that the complexity difference between models justifies the cascade approach.
