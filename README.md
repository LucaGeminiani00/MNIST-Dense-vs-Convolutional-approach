# MNIST-Dense-vs-Convolutional-approach
This directory applies a Dense and a Convolutional neural network to the same dataset (MNIST) for image recognition. 
The Convolutional approach is computationally heavier but achieves a better accuracy, surpassing the 99% threshold after 500 iterations. The dense implementation instead remains below 99% and achieves accuracies of around 98% after 500 epochs. 
The Convolutional approach is best implemented exploiting an NVIDIA GPU and Cuda.jl (I provide code also for this). 
