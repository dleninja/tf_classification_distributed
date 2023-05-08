# Tensorflow GPU Distribution Strategy

<p align="center">
<img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-ar21.png" alt="Alt text" width="600" height="310">
</p>

TensorFlow is an open-source machine learning framework developed by Google. It allows you to build and train machine learning models for a wide range of applications, from image recognition and natural language processing to reinforcement learning and time-series analysis.

One of the challenges of training large machine learning models is that they often require a significant amount of computation and memory. One way to address this challenge is to use multiple processing units, such as multiple CPUs or GPUs, to distribute the computation and memory usage across different devices. This is known as distributed training.

TensorFlow supports several strategies for distributed training, including the mirror distribution strategy. The mirror distribution strategy is a synchronous training strategy that replicates the model across multiple devices, such as multiple GPUs or multiple machines. Each device computes the forward and backward passes independently on its own subset of the data, and then communicates the gradients with the other devices to update the model weights. This communication between the devices introduces additional overhead, but it can allow you to train larger models or process more data than you can fit on a single device.

The mirror distribution strategy is particularly useful for deep learning models that require a significant amount of computation and memory. By using multiple devices, you can distribute the computation and memory usage across the available resources, reducing the training time and memory requirements for each device. However, it's important to carefully configure the strategy and tune the hyperparameters to achieve good performance and avoid issues such as communication bottlenecks or device synchronization problems.

Overall, the mirror distribution strategy in TensorFlow can be a powerful tool for scaling up machine learning models and achieving better performance and efficiency in distributed training.

## Types of Parallelism:

### `Data parallelism`

In data parallelism, the same model is replicated across multiple devices, such as GPUs or machines, and each device is responsible for processing a different subset of the input data. The outputs from all devices are then collected, combined, and used to update the model parameters. The goal is to speed up training by processing more data in parallel.

For example, let's say we have a dataset of images and we want to train a convolutional neural network (CNN) to classify them. We could use data parallelism to distribute the training across multiple GPUs. Each GPU would process a different batch of images and compute the gradients for that batch. The gradients would then be averaged across all GPUs and used to update the model weights.

### `Model parallelism`

In model parallelism, a single model is divided across multiple devices, and each device is responsible for computing a different part of the model's output. This approach is typically used for very large models that cannot fit in the memory of a single device.

For example, let's say we have a large neural network with millions of parameters, and we want to train it on a single GPU. However, the model is too large to fit in the GPU's memory. We could use model parallelism to split the model across multiple GPUs, with each GPU responsible for computing a different part of the model's output. The outputs from all GPUs would then be combined to produce the final output.

<p align="center">
<img src="https://github.com/dleninja/tf_classification_distributed/blob/main/misc/example_parallelism.png" alt="Alt text" width="600" height="310">
</p>

<p align="left">
<em> Illustration of differences between data and model parallelism. As the name suggests, in data parallelism, the model is replicated on multiple devices, e.g., GPUs, and training on different batches of image samples are computed. Whereas in model parallelism, the model is partitioned onto multiple devices, e.g., GPUs [1].</em>
</p>

#### Note:

At the time of this repository, Tensorflow only supports data parallelism, however Tensorflow has plans to implement model parallelism in the future.

## tf.distribute.MirroredStrategy

The tf.distribute.MirroredStrategy function is a TensorFlow API for synchronous distributed training. It allows you to distribute the training of a TensorFlow model across multiple GPUs or machines, improving performance and reducing the time required to train large models.

The basic idea of the `MirroredStrategy` is to replicate the model across multiple devices, such as GPUs or machines, and divide the data into multiple batches. Each device trains on a separate batch of the data and updates its own copy of the model parameters. Once all devices have finished training on their respective batches, the updates are collected, averaged, and applied to all copies of the model. This process is repeated for each batch of data until the entire dataset has been processed.

### Example code snippet

```
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = tf.keras.Sequential([...]) # define your model here

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32) # train the model
```

In this repository, I will demonstrate an example of using the `tf.distribute.MirroredStrategy` for a classification model using the MNIST dataset available through TensorFlow.

## Useful links:

## References/Links
* [1] https://docs.chainer.org/en/v7.8.0/chainermn/model_parallel/overview.html#model-parallelism
* https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy
