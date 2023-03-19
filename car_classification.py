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
	print('\n-------------------------------------------------------------------------------------\n')
	print('The array of true: ' + str(y_true))
	print('The array of predicted: ' + str(y_pred))

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
#	VGG16
################################################################################
def vgg16_func(train_im, train_labels, test_im, test_labels):

    # train_im = train_im.reshape(-1 , 28, 28)
    # test_im = test_im.reshape(-1,28,28)
    # train_im = tf.expand_dims(train_im, axis=3, name=None)
    # test_im = tf.expand_dims(test_im, axis=3, name=None)
    # train_im = tf.image.grayscale_to_rgb(train_im)
    # test_im = tf.image.grayscale_to_rgb(test_im)
    # train_im = tf.image.resize(train_im, [48, 48], method = 'bilinear')
    # test_im = tf.image.resize(test_im, [48, 48], method = 'bilinear')
    
    arr = [i for i in range(197)] 
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

    # This for loop freezes the convolution weights
    # for layer in model.layers:
        # layer.trainable = False
    model.summary()
    # x = layers.Flatten()(model.output)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)

    # predictions = layers.Dense(10, activation = 'softmax')(x)
    # head_model = Model(inputs = model.input, outputs = predictions)
        # This for loop freezes the fully connected layers
    # for layer in head_model.layers[:-1]:
    #     layer.trainable = False
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(train_im, train_labels,epochs = 5)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)



################################################################################
#	RESNET50
################################################################################
def resnet50(train_im, train_labels, test_im, test_labels):

    # train_im = train_im.reshape(-1 , 28, 28)
    # test_im = test_im.reshape(-1,28,28)
    # train_im = tf.expand_dims(train_im, axis=3, name=None)
    # test_im = tf.expand_dims(test_im, axis=3, name=None)
    # train_im = tf.image.grayscale_to_rgb(train_im)
    # test_im = tf.image.grayscale_to_rgb(test_im)
    # train_im = tf.image.resize(train_im, [48, 48], method = 'bilinear')
    # test_im = tf.image.resize(test_im, [48, 48], method = 'bilinear')
    
    arr = [i for i in range(197)] 
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)

    # This for loop freezes the convolution weights
    # for layer in model.layers:
        # layer.trainable = False
    model.summary()
    # x = layers.Flatten()(model.output)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    # x = layers.Dense(4096, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)

    # predictions = layers.Dense(10, activation = 'softmax')(x)
    # head_model = Model(inputs = model.input, outputs = predictions)
        # This for loop freezes the fully connected layers
    # for layer in head_model.layers[:-1]:
    #     layer.trainable = False
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(train_im, train_labels,epochs = 50)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)



################################################################################
#	AlexNet
################################################################################
def alexnet(train_im, train_labels, test_im, test_labels):

    # train_im = train_im.reshape(-1 , 28, 28)
    # test_im = test_im.reshape(-1,28,28)
    # train_im = tf.expand_dims(train_im, axis=3, name=None)
    # test_im = tf.expand_dims(test_im, axis=3, name=None)
    # train_im = tf.image.grayscale_to_rgb(train_im)
    # test_im = tf.image.grayscale_to_rgb(test_im)
    train_im = tf.image.resize(train_im, [227, 227], method = 'bilinear')
    test_im = tf.image.resize(test_im, [227, 227], method = 'bilinear')
    
    arr = [i for i in range(197)] 
    # # (3) Create a sequential model
    # model = tf.keras.Sequential()

    # # 1st Convolutional Layer
    # model.add(layers.Convolution2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),strides=(4,4), padding='valid'))
    # model.add(layers.Activation('relu'))
    # # Pooling 
    # model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # # Batch Normalisation before passing it to the next layer
    # model.add(layers.BatchNormalization())

    # # 2nd Convolutional Layer
    # model.add(layers.Convolution2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    # model.add(layers.Activation('relu'))
    # # Pooling
    # model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # 3rd Convolutional Layer
    # model.add(layers.Convolution2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    # model.add(layers.Activation('relu'))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # 4th Convolutional Layer
    # model.add(layers.Convolution2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    # model.add(layers.Activation('relu'))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # 5th Convolutional Layer
    # model.add(layers.Convolution2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    # model.add(layers.Activation('relu'))
    # # Pooling
    # model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # Passing it to a dense layer
    # model.add(layers.Flatten())
    # # 1st Dense Layer
    # model.add(layers.Dense(4096, input_shape=(224*224*3,)))
    # model.add(layers.Activation('relu'))
    # # Add Dropout to prevent overfitting
    # model.add(layers.Dropout(0.4))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # 2nd Dense Layer
    # model.add(layers.Dense(4096))
    # model.add(layers.Activation('relu'))
    # # Add Dropout
    # model.add(layers.Dropout(0.4))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # 3rd Dense Layer
    # model.add(layers.Dense(1000))
    # model.add(layers.Activation('relu'))
    # # Add Dropout
    # model.add(layers.Dropout(0.4))
    # # Batch Normalisation
    # model.add(layers.BatchNormalization())

    # # Output Layer
    # model.add(layers.Dense(17))
    # model.add(layers.Activation('softmax'))

    # model.summary()

    # # (4) Compile 
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
        layers.Dense(10, activation='softmax')
    ])





    # (5) Train
    model.fit(train_im, train_labels,epochs = 50)
    predictions_arr = model.predict(test_im)
    prediction= [arr[np.argmax(predictions_arr[i])] for i in range(test_im.shape[0])]
    return(prediction, test_labels)


################################################################################
#	Load and Test Data
################################################################################

# Trains a classifier of the specified type on the MNIST digit dataset
def car_classifier(t, n, m):
    
    training_dataset = h5py.File('./train_cars.h5', "r")
    testing_dataset = h5py.File('./test_cars.h5', "r")

    train_im = np.array(training_dataset["dataset_x"][:])
    train_labels = np.array(training_dataset["dataset_y"][:])
    print(np.max(train_labels))
    test_im = np.array(testing_dataset["dataset_x"][:])
    test_labels = np.array(testing_dataset["dataset_y"][:])
    # train_im = np.true_divide(train_im, 255)
    # test_im = test_im/255
    
    results = []
    
    
    if(t == "resnet50") : results.append(resnet50(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    elif(t == "vgg16") : results.append(vgg16_func(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    elif(t == "alexnet") : results.append(alexnet(train_im[0:n], train_labels[0:n], test_im[0:m], test_labels[0:m]))
    arr = [i for i in range(197)] 
    for pred, true in results : Confusion_Matrix(pred, true, arr)
    # Load Dataset (note that test data is loaded separately lower down - this helps with memory)
    # ds_train, ds_info_train = tfds.load(name="cars196", split="train", with_info=True)
    # fig = tfds.show_examples(ds_train, ds_info_train)
    # ds_test, ds_info_test = tfds.load(name="cars196", split="test", with_info=True, as_supervised=True)
    # fig2 = tfds.show_examples(ds_test, ds_info_test)
    
    # train_images = tfds.as_numpy(ds_train)
    # test_images = tfds.as_numpy(ds_test)
    
    # train_im = []
    # train_labels = []
    
    
    # for ex in train_images:
    #     train_im.append(ex['image'])
    
    # train_im = np.array(train_im)
    # print(train_im.shape())
    # # train_images = np.array(train_images)
    # # train_labels = np.array(train_labels)
    # # test_images = np.array(test_images)
    # # test_labels = np.array(test_labels)

    # print ("train_images shape: " + str(train_images.shape))
    # print ("train_labels shape: " + str(train_labels.shape))
    # print ("test_images shape: " + str(test_images.shape))
    print ("test_images shape: " + str(test_labels.shape))
	# Load an norm data
    return 0








################################################################################
#	Main
################################################################################
def main():
    #Max is 60,000 samples of training dat
    #Max is 10,000 samples of testing data
    training_samples = 8144
    testing_samples = 8041
    # MNIST_classifier( "vgg16_scratch" , training_samples, testing_samples)
    car_classifier( "alexnet" , training_samples, testing_samples)
    car_classifier( "resnet50" , training_samples, testing_samples)
    car_classifier( "vgg16" , training_samples, testing_samples)
 
if __name__ == "__main__":
    main()