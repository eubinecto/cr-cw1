# Coursework on Deep Learning and Robotics (COMP34212)
- author: Eu-Bin KIM (10327741)

## 1. Introduction

This report aims to discuss the state-of-the-art of deep learning in cognitive robotics, and demonstrate how we can
train & optimise a CNN to classify images of Cifar10. A literature review of the application of DNN's, the pros and cons of DNN's in the context of cognitive
robotics is detailed in section 2. We then train a CNN model to classify images of Cifar10, while carrying out a number of 
experiments to optimise the model in section 3-4. 

## 2. Literature review

Over the last decade, the application of deep learning has surged in many domains of cognitive Robotics.
Deep learning models have been actively used, and proved to work well in cognitive robotics to detect & perceive objects, to manipulate objects and to handle multimodal
data, just to name a few (Pierson & Gashler, 2017).
For instance, Mariolis et al. used (2015) deep learning to detect the grasping point of garments that are grasped by a robotic gripper. Their deep learning
approach outperformed a state-of-the-art model with the mean error of 5.3cm.
 Lenz et al. used (2015) deep learning to detect optimal grasps for manipulating objects. Their model were able to grasp objects with 84-89% success rate,
which was phenomenal compared with a 31% rate of a state-of-the-art model.
Schmitz et al. used (2014) deep learning to detect objects with multimodal features, such as signals from skin sensors, 
fingertip forces, joint torques, actuator currents, etc. Their model achieved 88% accuracy, which was again far better
than 68% of a previously used rule-based approach model.


Why is deep learning working so well in cognitive robotics? one of the reason
is that DNN models automatically extract hidden features from raw data, without the need of domain-specific feature engineering (Pierson , 2017).
This property is especially useful in cognitive robotics, where the data often consist of a set of signals that are hard to interpret even for 
a human expert (e.g. a combination of fingertip forces, joint torques, actuator sensors). Another reason DNN works well
might be because they adapt. That is, a deep learning system can automatically learn new situations (Sunderhauf, 2018). As opppsed to
rule-based models, DNN's can change their "rule" by simply updating their weights with respect to newly observed data. 


DNN's do come with a limit. Sunderhauf et al. and Pierson & Gashler unanimously point out that DNN models require a large amount of data.
This is especially limiting in cognitive robotics because it is costly to generate training data on physical systems (Pierson & Gashler, 2017).  
This however could be mitigated by using simulation to generate a set of virtual data.
  
## 3. The methods

Convolutional Neural Networks (CNN's) are the kinds of DNN's that are used most popularly to detect objects in cognitive robotics. In the following
section, we demonstrate how we could go about training a simple version of CNN and optimising its architecture & hyper parameters.

![The architecture of RegThreeCNN.](.report_images/a193ff20.png){width=300px}

We first have a look at the final product. **Figure 1** shows the architecture of the final model, namely RegThreeCNN.
It is trained on zero-centered Cifar10 images over 17 epochs. It is composed of three convolution layers of equal dimensions, each of which consists of a Batch Normalization layer, a Max Pool layer and a Dropout layer.
The model is implemented with Pytorch, as the author is more familiar with Pytorch than Keras. 

Four experiments have been carried out to come up with RegThreeCNN. We find the optimal epoch (exp 1), 
zero-center the image data (exp 2), optimise the number of convolution layers (exp 3) and regularise the network (exp 4).


## 4. Results & Discussions
### 4.1. Experiment 1: Optimising epochs


In Experiment 1, we come up with a way of optimising epochs: Early Stopping. 
Optimising epochs is crucial because, if we set it too few/too much, the model will underfit/overfit to
the training data. Early Stopping is a simple technique we can use for optimising epochs; 
we set aside a validation set, evaluate the validation loss at the end of each epoch, and stop the training when the validation loss no longer improves (Geron, 2019).
For instance, we see that the optimal epoch for `BaseCNN` is 4, because the validation loss no longer improves after the fourth epoch (**Figure 2**).
This is a simple yet effective way of optimising epoch, and hence we apply this technique to all the experiments below.

![The training loss and validation loss of BaseCNN over 7 epoch.](.report_images/0b4ae5fb.png){width=300px}


### 4.2. Experiment 2: Zero-centering the image data

model | image data (`[0, 1]`) | image data (`[-1, 1]`) | epoch
--- | --- | --- | ---
BaseCNN | 61.29%  | **63.12%** | 4

Table: The testing accuracies of BaseCNN model trained on unscaled and scaled image data. The number of
epoch is optimised with Early Stopping.

From Experiment 2, we see that zero-centering the image data results in an increase in accuracy. 
In experiment 2, we rescale the original image data of `[0, 1]` range to those of `[-1, 1]` range.
We do this to avoid confining the direction of the gradient descent step. When the input to each neuron is always positive (e.g. features of `[0, 1]` range),
then the local gradients with respect to the neuron's weights will always be positive (since $\frac{df}{dw_{i}} = \sum x_{i}$ for $f = \sum w_{i} x_{i} + b$).
This means that the global gradients for the weights will be either all positive or all negative depending on the upstream gradient, which leads to
the gradient descent taking zig-zag steps towards the optimum (Karpathy, 2016). We can mitigate this inefficiency by transforming feature values to a mixture of positive and negative values.
As such, we get 2% increase in accuracy with BaseCNN when we train it on the image data rescaled to `[-1, 1]` range (**Table 1**).


### 4.3. Experiment 3: Optimising the number of convolution layers

model | training accuracy | testing accuracy | epoch   
--- | --- | --- | --- 
BaseCNN | 75.94% | 62.12% | 4
TwoCNN |  74.42% | **67.36%** | 8
ThreeCNN | 72.02% | 66.41% | 8

Table: The training and testing accuracies of the CNN's with different numbers of convolution layers.
The number of epoch is optimised with Early Stopping.

From Experiment 3, we see that carelessly adding more layers does not lead to improved performance.
With increased number of convolutional layers, we can expect the model to perform better as it could be capable of extracting 
more features and generalise better. This is why TwoCNN significantly outperforms BaseCNN by 5% in accuracy (**Table 2**),
although its training accuracy is lower than that of BaseCNN's. However, ThreeCNN performs
worse than TwoCNN by 1% in accuracy despite having more layers. This is potentially due to internal covariate shift;
It is difficult to optimise deep neural nets because the input from prior layers can change after weight updates (Brownlee, 2019).


### 4.4. Experiment 4: Regularising the network


model |training accuracy | testing accuracy | epoch
--- | --- | --- | ---
RegBaseCNN | 71.155% | 59.73% | 3
RegTwoCNN | 76.6625% | 69.3% | 9 
RegThreeCNN | 78.78% | **71.2%** | 17

Table: The training and testing accuracies of the regularised CNN's.  The number of epoch for each model
is optimised with Early Stopping.

From experiment 4, we see that adding Batch Normalisation and Dropout layers makes it easier to train deep models.
With Batch Normalisation, we can solve the problem of internal covariate shift, and thereby regularise the model as well (Brownlee, 2019). 
Dropout can also be used to regularise the model, as it "turns off" potentially wasteful neurons (Srivastava et al., 2014). With the two techniques applied,
we see that we can now benefit from more layers. In contrast to those we saw in **Table 2**, both the training and testing accuracies 
in **Table 3** steadily increases with more layers, with RegThreeCNN being the deepest and best performing model of all (71.2%).


### 4.5. Future Work

We could improve upon RegThreeCNN either by carefully extending Vanilla CNN or adopting new architectures. 
We could try extending RegThreeCNN to implement VGG-16, which is a model solely made of a series of convolution, Max Pooling
and fully connected layers (Simonyan et al., 2014). Despite its simplicity, it achieves up to 92 accuracy in Cifar10 dataset.
We could also try implementing ResNet by adding Residual Blocks, which leverages skip connects to better train deep models (He et al., 2015). 
Lastly, we could also try a completely different model, a Vision Transformer, which is a Transformer adapted to classify images (Dosovitskiy et al., 2020).


## 5. References
- Brownlee, J. 2019. *A Gentle Introduction to Batch Normalization for Deep Neural Networks*. Available at: https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
- Dosovitskiy, A. et al. 2020. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale 2020*
- He, K. et al., 2015.  *Deep Residual Learning for Image Recognition*
- Karpathy, A. 2016. CS231n Convolutional Neural Networks for Visual Recognition 2016 - Standford University. *Lecture 6:Traning NN 1*.
- Lenz I, Lee H, Saxena A. 2015. *Deep learning for detecting robotic grasps*. Int J Robotics res. 2015;34(4–5):705–724.
- Mariolis I, Peleka G, Kargakos A, et al. 2015. *Pose and category recognition of highly deformable objects using deep learning*. In: 2015 International Conference on Advanced Robotics (ICAR); Istanbul; 2015. p. 655–662.
- Pierson & Gashler, H. Gashler, M. 2017. *Deep learning in robotics: a review of recent research*. 
- Schmitz A, Bansho Y, Noda K, et al. *Tactile object recognition using deep learning and dropout*. 14th IEEE-RAS International Conference on Humanoid Robots; Madrid, Spain; 2014. p. 1044–1050.
- Simonyan et al. 2014. *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
- Srivastava, N. et al. 2014. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
- Sunderhauf, N. et al., 2017. *The limits and potentials of deep learning for robotics*.
- Geron, A. 2019. *Hands-on machine learning with Scikit-Learn, Keras and TensorFlow: concepts, tools, and techniques to build intelligent systems (2nd ed.)*. O’Reilly.  
