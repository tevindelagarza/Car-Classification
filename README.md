# Final Project Proposal: Car Classification Author: Tevin De La Garza

The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images:
http://ai.stanford.edu/~jkrause/cars/car_dataset.html


![image](https://user-images.githubusercontent.com/62819751/220127673-d697cd06-2106-4f6b-ae17-987041e6d675.png)


The purpose of this project would be to use a supervised learning algorithm to determine the class of the car. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. A supervised learning approach would be best suited for this type of problem as there is training data with labels and
testing data with labels. 

The data will be loaded into the program by downloading the dataset and feeding it into an Hierarchical Data Format (HDF) file namely an h5 file. It contains multidimensional arrays of scientific data that will help get the car data into an appropiate format to train. The images will be loaded into the h5 file along with the labels. The data will then be split up in the code to training_images and training_labels. 

A confusion matrix will be generated to determine how accurate the model was and whether the prediction was true or not to determine the class. 

If there is time I would like to load more images of cars that the dataset does not have like subarus, so the algorithm can determine more cars and it will help further prove that is classifying the cars correctly. 


RESNET50
------------------------------------------------------------------------------------------

The attached resnet50modelsummary.txt shows the model summary.


Results
------------------------------------------------------------------------------------------


VGG16
------------------------------------------------------------------------------------------



Results
------------------------------------------------------------------------------------------
