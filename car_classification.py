import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds
#from keras.datasets import cars196
import numpy as np
import seaborn as sns
import collections
import h5py
import sklearn.metrics as met
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model

################################################################################
#	Confusion Matrix and Accuracy
################################################################################

# Calculates and plots the confusion matrix
def Confusion_Matrix(y_pred, y_true, y_classes):
	# print('\n-------------------------------------------------------------------------------------\n')
	# print('The array of true: ' + str(y_true))
	# print('The array of predicted: ' + str(y_pred))

	#Display the accuracy of the algorithm
	print('\n-------------------------------------------------------------------------------------\n')
	Accuracy = (y_pred == y_true).sum() / len(y_pred) 
	print('Total Accuracy: ' + str(Accuracy))
	print('\n-------------------------------------------------------------------------------------\n')

	# Create confusion matrix with the true value vs the predicted value
	confusion_matrixs = met.confusion_matrix(y_true, y_pred, labels=y_classes)
	cm_display = met.ConfusionMatrixDisplay(confusion_matrix = confusion_matrixs, display_labels = y_classes)
	cm_display.plot()
	plt.show()



################################################################################
#	VGG19
################################################################################
def vgg19_func(train_im, train_labels, test_im, test_labels):

    arr = [i for i in range(197)] 
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=True)

    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(train_im, train_labels,epochs = 6)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)



################################################################################
#	RESNET50V2
################################################################################
def resnet50v2(train_im, train_labels, test_im, test_labels):

    arr = [i for i in range(197)] 
    model = tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet', include_top=True)
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(train_im, train_labels,epochs = 20)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)


################################################################################
#	RESNET50
################################################################################
def resnet50(train_im, train_labels, test_im, test_labels):
    
    arr = [i for i in range(197)] 
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(train_im, train_labels,epochs = 20)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)

################################################################################
#	AlexNet
################################################################################
def alexnet(train_im, train_labels, test_im, test_labels):

    
    arr = [i for i in range(197)] 


    model = tf.keras.Sequential([
        layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(197, activation='softmax')
    ])



    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    model.summary()

    model.fit(train_im, train_labels,epochs = 50)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)


################################################################################
#	Load and Test Data
################################################################################

# Trains a classifier of the specified type on the MNIST digit dataset
def car_classifier(t, n, m):
    if(t == "alexnet") :
        training_dataset = h5py.File('./train_cars_227.h5', "r")
        testing_dataset = h5py.File('./test_cars_227.h5', "r")

        train_im = np.array(training_dataset["dataset_x"][:])
        train_labels = np.array(training_dataset["dataset_y"][:])
        test_im = np.array(testing_dataset["dataset_x"][:])
        test_labels = np.array(testing_dataset["dataset_y"][:])
    else :
        training_dataset = h5py.File('./train_cars_224.h5', "r")
        testing_dataset = h5py.File('./test_cars_224.h5', "r")

        train_im = np.array(training_dataset["dataset_x"][:])
        train_labels = np.array(training_dataset["dataset_y"][:])
        test_im = np.array(testing_dataset["dataset_x"][:])
        test_labels = np.array(testing_dataset["dataset_y"][:])

    
    results = []
    
    
    if(t == "resnet50v2") : results.append(resnet50v2(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    elif(t == "resnet50") : results.append(resnet50(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    elif(t == "vgg19") : results.append(vgg19_func(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    elif(t == "alexnet") : results.append(alexnet(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    arr = [i for i in range(197)] 
    for pred, true in results : Confusion_Matrix(pred, true, arr)

    return 0








################################################################################
#	Main
################################################################################
def main():
    #Max is 60,000 samples of training dat
    #Max is 10,000 samples of testing data
    training_samples = 8144
    testing_samples = 8041

    # car_classifier( "resnet50" , training_samples, testing_samples)
    #car_classifier( "resnet50v2" , training_samples, testing_samples)
    # car_classifier( "vgg19" , training_samples, testing_samples)
    car_classifier( "alexnet" , training_samples, testing_samples)
if __name__ == "__main__":
    main()