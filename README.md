# Final Project Proposal: Car Classification Author: Tevin De La Garza

The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images:
http://ai.stanford.edu/~jkrause/cars/car_dataset.html


![image](https://user-images.githubusercontent.com/62819751/220127673-d697cd06-2106-4f6b-ae17-987041e6d675.png)


The purpose of this project would be to use a supervised learning algorithm to determine the class of the car. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. A supervised learning approach would be best suited for this type of problem as there is training data with labels and
testing data with labels. 

The data will be loaded into the program by downloading the dataset and feeding it into an Hierarchical Data Format (HDF) file namely an h5 file. It contains multidimensional arrays of scientific data that will help get the car data into an appropiate format to train. The images will be loaded into the h5 file along with the labels. The data will then be split up in the code to training_images and training_labels. 

A confusion matrix will be generated to determine how accurate the model was and whether the prediction was true or not to determine the class. 

If there is time I would like to load more images of cars that the dataset does not have like subarus, so the algorithm can determine more cars and it will help further prove that is classifying the cars correctly. 

h5.py
------------------------------------------------------------------------------------------
The h5.py takes in the dataset. The dataset is divided into folders of car training images and car testing images. The labels are found in the devkit and are assigned numbers to each of the labels (Make, Model, Year). The h5.py file compresses these files and resizes them to match what is needed for the CNN to be processed. It reads in the images individually and resizes them to 224x224x3 for the RESNET and VGG19 using opencv. It also adjusts the colors to range from 0-1 instead 0-255 for normalization. For the Alexnet Model it was adjusted to 227x227x3 to cater for that model. After the tool was ran it produced the h5 files:

test_cars_224.h5

train_cars_224.h5

test_cars_227.h5

train_cars_227.h5

These files contains the data array that will be used and has the labels associated with each image. 


RESNET50 and RESNET50V2
------------------------------------------------------------------------------------------
RESNET stands for Residual Network and is a convolution neural network that was introduced in "Deep Residual Leaning for Image Recognition" by He Kaimin, Zhang Xiangyu, Ren Shaoqing, and Sun Jian. This has 50 layers of convulational neural network (48 conv, 1 MaxPooling, and 1 Avg pool).

The attached ResNet50_Model_Summary.txt shows the model summary.

The attached ResNet50V2_Model_Summary.txt shows the model summary.

The main difference between ResNet50 and ResNet50V2 is that using a 1001-layer ResNet on CIFAR-10 (4.62% error) and CIFAR-100, and a 200-layer ResNet on ImageNet improved the results.


RESNET50:
https://arxiv.org/abs/1512.03385 

RESNET50V2:
https://arxiv.org/abs/1603.05027


<img width="991" alt="image" src="https://user-images.githubusercontent.com/62819751/226486819-446f08f2-d963-4aba-bd1e-2d158031db6e.png">

The Image above shows the network and how it processes the images until it uses the softmax function at the end to make the final prediction.


Results
------------------------------------------------------------------------------------------
RESNET50:

<img width="394" alt="Screenshot 2023-03-20 092625" src="https://user-images.githubusercontent.com/62819751/226486681-16fc0685-d7fb-49bf-85be-925bec697018.png">

<img width="433" alt="resnet" src="https://user-images.githubusercontent.com/62819751/226486449-36076515-5b8b-49cf-ba13-7a4f95e44948.png">


RESNET50V2:

<img width="394" alt="Screenshot 2023-03-20 092625" src="https://user-images.githubusercontent.com/62819751/226486690-ba502a32-46b6-4a28-bd81-0d32af33b2b9.png">

<img width="171" alt="Screenshot 2023-03-20 092640" src="https://user-images.githubusercontent.com/62819751/226486492-b3b62973-1f35-4b71-9deb-e56023b2dcd7.png">




VGG19
------------------------------------------------------------------------------------------
VGG19 is a variant of VGG model which in short consists of 19 layers (16 convolution layers, 3 Fully connected layer, 5 MaxPool layers and 1 SoftMax layer).

A fixed size of (224 * 224) RGB image was given as input to this network which means that the matrix was of shape (224,224,3).

<img width="960" alt="image" src="https://user-images.githubusercontent.com/62819751/226488809-2bd0d1ba-1262-4215-9908-197f10f5aeb9.png">

The above image clearly shows how vgg19 architecture works.

It takes the image and uses convolution and maxpooling layers until it reaches the fully connected layers and then the softmax layer for final decision making. 

vgg19_Model_Summary.txt shows the model summary.



Results
------------------------------------------------------------------------------------------
<img width="377" alt="Screenshot 2023-03-20 165219" src="https://user-images.githubusercontent.com/62819751/226486696-ecda713f-d433-4e97-9584-5a46c38c5790.png">

<img width="182" alt="Screenshot 2023-03-20 165202" src="https://user-images.githubusercontent.com/62819751/226486521-d7733e33-5857-4234-8554-40d86db76325.png">



AlexNet
------------------------------------------------------------------------------------------
Unfortunately, everytime I tried to run the AlexNet Model my computer ran out of memory during the training layer. Next time I need to use Keras with the GPU but VScode was giving trouble with anaconda to make that happen in a timely manner. It is possible and there was documentation but it was a little out of date. 

Anyways, AlexNet_Model_Summary.txt shows the AlexNet Model.

<img width="985" alt="image" src="https://user-images.githubusercontent.com/62819751/226489246-cbe624f5-4a2b-4fc6-83bd-f2650cec2888.png">


Final Thoughts
------------------------------------------------------------------------------------------

A little dissapointed that the results were not stellar as anticipated.

The results were not too great even with a longer training session of 20 epochs. This is due to the images not being preprocessed and augmented. If there was more time I would have used KOALAs image augmentation to augment the images for better results.

https://medium.com/analytics-vidhya/image-augmentation-9b7be3972e27

Image data augmentation is used to expand the training dataset to improve the model’s performance and ability to generalize. This would have gave greater accuracy when predicting what car it could have been.


Regardless, the model was able to be processed the CNN and was able to train and classify the images. It was a fun experiment that could be done better with a faster machine and more time for training the models with more data augmentations. 


