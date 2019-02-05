import { layers, sequential, Model } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

export const buildModel = (): Model => {
  const model = sequential();

  // First convolutional layer, receive 28x28x1 inputs, applies 8 filters
  model.add(
    layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    })
  );

  // First Max pooling
  model.add(
    layers.maxPool2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  // Second convolutional layer, applies 16 layers
  model.add(
    layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    })
  );

  // Second Max Pooling
  model.add(
    layers.maxPooling2d({
      poolSize: [2, 2],
      strides: [2, 2]
    })
  );

  // Flattens from 2d to 1d
  model.add(layers.flatten());

  // Fully connected layer
  model.add(
    layers.dense({
      units: 64,
      kernelInitializer: 'VarianceScaling',
      activation: 'relu'
    })
  );

  // Fully connected layer
  model.add(
    layers.dense({
      units: 10,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    })
  );

  return model;
};
