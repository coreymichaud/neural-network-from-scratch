# Neural Network From Scratch
I created a neural network from scratch in Python using only numpy to better understand how a neural network works. In this project, I specifically created a multilayer perceptron to classify a handwritten number from the Modified National Institute of Standards and Technology (MNIST) dataset.

## Motivation
I have taken many machine learning classes; some where I learned how to code a model, and others where I learned the math behind it. I have worked on a few neural networks before, but never from scratch (without the use of a major NN library). The goal of this project was to obtain a better understanding of how a neural network works from the training, to the math behind the algorithms, to the activation function used. 

## Results

You can review my code to see exactly how it works, but the results from running the model are:
![MNIST_results](https://github.com/coreymichaud/Neural-Network-From-Scratch/assets/63071835/d6cd6784-dc6d-4dc9-88b9-93fc310f35fb)

After 20 epochs, the training accuracy was 0.9943, while the test accuracy was 0.9646. This multilayer perceptron resulted in very high training accuracy, with slightly lower test accuracy, meaning it overfit the data slightly, but for something this simple, I'll take it. Possibly in the future I will modify the code to try to obtain better test accuracy and show results with different metrics.
