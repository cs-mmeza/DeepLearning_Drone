**Content/index**
- [Introduction](#introduction)
- [Network Architecture](#network-architecture)
- [Hyperparameters and more](#hyperparameters-and-more)
  - [Optimizer](#optimizer)
- [Results](#results)
- [Future Enchantments](#future-enchantments)
- [Sources](#sources)



##### Notes & Specifications:

This project was trained with a local machine.

<b>CPU:</b> Ram-16gb, Ryzen 1700, Nvidia GTX 1080 GPU

The training process took me about 1 hour and 30 min. 
The processing speed for this GPU is good enough, and it can achieve a better result with experimentation.

#### Introduction

The main objective of this project is to apply a Fully Convolutional Neural Network (FCNN) in a quad-drone. The drone identifies a person with specific features (red t-shirt in this case). Recognition is possible by training the FCNN with images from a data set. The data input comes from a drone that captures images from simulated open space.

The project runs in a unity simulation, Where it is possible to control a quad-drone that can be moved freely in the simulated open space. I set up a pattern that the drone will follow to capture images for recognition. After that, I set up spawning points where the multitude appears. The objective appears in the simulated crowd, which corresponds to the noise in the model. More details are in the image below. <em>(Figure 1, Figure 2)</em>

<p align = "center">
<img src = "/Udaciy_ROS_ND/RoboND-DeepLearning-Project-master/docs/misc/Figure1.PNG">
</p>
<p align = "center">
Fig 1, <em>Spawn points(pink and blue) and patterns(green) that spawn people and move the drone on patrol mode.</em>
</p>

<p align = "center">
<img src = "/Udaciy_ROS_ND/RoboND-DeepLearning-Project-master/docs/misc/Figure2.PNG">
</p>
<p align = "center">
Fig 2, <em> Running and collecting data.</em>
</p>

The quad-drone can record and save images. Those images will be pre-processed and used as inputs for our FCNN. The model performance is measured with IOU (Intersection Over Union Metric), Which takes a set of pixels (AND intersection) and divides it by the union set of the same pixels (OR union).

This model has been trained to recognize people. Thus, the model can't be used to predict other targets in different scenarios (cats, cars, dogs, etc.). To use the model for other applications, We need to feed and train it with the corresponding data.

<p align = "center">
<img src = "/Udaciy_ROS_ND/RoboND-DeepLearning-Project-master/docs/misc/Figure3.png">
</p>
<p align = "center">
Fig 3, <em> Target of this project.</em>
</p>

#### Network Architecture

The system is composed of an encoder-decoder FCNN. Encoding and decoding are frequently used to train deep learning systems, Because of the efficiency that provides. A brief explanation of their function is that the encoder takes the input, which in this case, are the images taken by the drone, and processes the information, reducing the spatial dimension. This data simplification is made to learn specific objects or targets in the images. Thus, instead of processing the whole set of pixels, the model can select the small parts of the data, gathering more details from the dataset. Then we use this specific information that we obtain to build a classification. The encoder also contains a pooling sampling which down-samples the data. The pooling sampling is used to generalize the images we have not seen before and reduced the risk of over-fitting. 

The decoded sections gradually recover the size of the output data. Then the data is transformed again to the same size as the input image. This is possible by mapping the low-resolution encoder feature to full input resolution maps for pixel-wise classification. We can see the architecture in the diagram below. 


<p align = "center">
<img src = "/Udaciy_ROS_ND/RoboND-DeepLearning-Project-master/docs/misc/diagram1.png">
</p>
<p align = "center">
Diagram 1, <em> FCNN Architecture.</em>
</p>

```python
def fcn_model(inputs, num_classes):
    

    # Add Encoder Blocks. 
    num_filters = 32
    encoder_layer_1 = encoder_block(inputs, num_filters, 2)
    encoder_layer_2 = encoder_block(encoder_layer_1, 2*num_filters, 2)
    encoder_layer_3 = encoder_block(encoder_layer_2, 4*num_filters, 2)
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    mid_layer = conv2d_batchnorm(encoder_layer_3, 4*num_filters ,kernel_size=1,strides=1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder_layer_1 = decoder_block(mid_layer, encoder_layer_2, 4*num_filters)
    decoder_layer_2 = decoder_block(decoder_layer_1, encoder_layer_1, 2*num_filters)
    decoder_layer_3 = decoder_block(decoder_layer_2, inputs, num_filters)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    outputs = layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder_layer_3)

    # print neural_network shapes
    print(inputs)
    print(encoder_layer_1)
    print(encoder_layer_2)
    print(encoder_layer_3)
    print(mid_layer)
    print(decoder_layer_1)
    print(decoder_layer_2)
    print(decoder_layer_3)
    print(outputs)
    
    return outputs
```

```python
image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)
```

<em>Output:</em>
```
Tensor("input_1:0", shape=(?, 160, 160, 3), dtype=float32)
Tensor("batch_normalization/batchnorm/add_1:0", shape=(?, 80, 80, 32), dtype=float32)
Tensor("batch_normalization_2/batchnorm/add_1:0", shape=(?, 40, 40, 64), dtype=float32)
Tensor("batch_normalization_3/batchnorm/add_1:0", shape=(?, 20, 20, 128), dtype=float32)
Tensor("batch_normalization_4/batchnorm/add_1:0", shape=(?, 20, 20, 128), dtype=float32)
Tensor("batch_normalization_6/batchnorm/add_1:0", shape=(?, 40, 40, 128), dtype=float32)
Tensor("batch_normalization_8/batchnorm/add_1:0", shape=(?, 80, 80, 64), dtype=float32)
Tensor("batch_normalization_10/batchnorm/add_1:0", shape=(?, 160, 160, 32), dtype=float32)
Tensor("conv2d_2/truediv:0", shape=(?, 160, 160, 3), dtype=float32)
```

The FCNN is composed of 3 encoder layers and 3 decoder layers. The encoder sections contain the pooling 
technique, that makes each layer lose information, but if we use “skip connections” from the encoder layer to 
the decoder layer decreases the loss of information.

To connect the encoder to the decoder layer instead of a fully connected layer we use a 1x1 convolutional layer 
to retain the spatial information. Once we processed these sections with the “1x1 convolution” we can up-sample 
the data in the decoder, by using bilinear interpolation.

Bilinear interpolation is a resampling technique that utilizes the weighted average of four nearest 
known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value.

#### Hyperparameters and more
##### Optimizer

```python
learning_rate = 0.001
batch_size = 64
num_epochs = 30
steps_per_epoch = 200
validation_steps = 50
workers = 2 
```
 
I used an Adam optimizer. We have some parameters that we can change to obtain better results in certain circumstances.

Batch size: 
Computing the gradient over the entire dataset is expensive and slow. Therefore, we can use batches. I choose to train the model with sub-sets of 64 batches, and I consider that a sub-set between 64 and 128 batches is good to build a model on a personal computer. This might decrease the accuracy of our project, but we can run it on our computer for sure. 

At first, it is better to keep things simple to make sure the overall network implementation was done correctly and can train. It is always possible to increase the depth and complexity of the network later if it runs fine. 

In general, a low learning rate can be the best option. I choose 0.001 as the learning rate. At the time, I did the test. 

Steps per epoch: 
The number of steps (batches of samples) before declaring your epoch finished. This should be the number of training images over batch size because you should theoretically train your entire data on many epochs. 

Validation Steps per epoch: 
This should be the number of validation images over batch size because you should test all your data on every epoch. 

Workers: 
This is the number of parallel processes during training. This can affect your training speed and it depends on your hardware. I tried a worker number of 2 for this test.

#### Results

<p align = "center">
<img src = "/Udaciy_ROS_ND/RoboND-DeepLearning-Project-master/docs/misc/Plot1.png">
</p>
<p align = "center">
Plot 1, <em> Training curves from the sixth epoch(top) to the last one(bottom-30).</em>
</p>

```python
# Sum all the true positives, etc from the three datasets to get a weight for the score
true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
print(weight)
```
<em>Output:</em>  `0.7508610792192881`

```python
# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU = (iou1 + iou3)/2
print(final_IoU)
```

<em>Output:</em> `0.542268221478`
```python
# And the final grade score is 
final_score = final_IoU * weight
print(final_score)
```
<em>Output:</em>` 0.407168102005`

Accuracy 0.407 is enough for the drone to follow the objective into open space and recognize it among the people. 


#### Future Enchantments

More layers should generate a deeper network. This would enable de FCNN to learn with smaller details. This Also might improve identifying the target from a long distance. 

Removing the images that don't contain people from the training data could improve the data received for the classification. This happens because training a model with the dataset, is skewed or unbalanced most of the time, and the classifications would bias towards the class with more data.

#### Sources
This project is part of the [Udacity Robotics Software Nanodegree program](https://www.udacity.com/enterprise/autonomous-systems). 
    
Skip connections:
https://www.youtube.com/watch?time_continue=3&v=JUYLA5PWzo0
Adam optimizer:
https://keras.io/api/optimizers/adam/