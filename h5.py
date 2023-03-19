import scipy.io 
import numpy as np
import h5py
import cv2 # install opencv-python-headless




LOCAL_SOURCE='/root/Car-Classification'
OUTPUT_PATH='/root/Car-Classification'



HEIGHT = 224
WIDTH = 224
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)

h5_train_proc = { "fname": "train_cars.h5", "imgDir": "cars_train", "matFile": "cars_train_annos.mat"}
h5_test_proc = { "fname": "test_cars.h5", "imgDir": "cars_test", "matFile": "cars_test_annos_withlabels.mat"}

for test_train in [h5_train_proc, h5_test_proc]:
  
  cars_annos = scipy.io.loadmat( LOCAL_SOURCE + '/devkit/{}'.format(test_train.get('matFile')) )
  
  car_bbox_x1 = np.zeros(0,)
  car_bbox_x2 = np.zeros(0,)
  car_bbox_y1 = np.zeros(0,)
  car_bbox_y2 = np.zeros(0,)
  car_class = np.zeros(0,)
  car_fname = np.zeros(0,)
  
  NUM_IMAGES = len(cars_annos['annotations'][0])
  
  car_image = np.zeros((NUM_IMAGES, HEIGHT, WIDTH, CHANNELS))
  
  # Iterating through the annotations and loading images and labels
  i = 0
  for car in cars_annos['annotations'][0]:
    car_bbox_x1 =  np.append(car_bbox_x1, car[0][0].item())
    car_bbox_x2 = np.append(car_bbox_x2, car[1][0].item())
    car_bbox_y1 = np.append(car_bbox_y1, car[2][0].item())
    car_bbox_y2 = np.append(car_bbox_y2, car[3][0].item())
    car_class = np.append(car_class, car[4][0].item())
    car_fname = np.append(car_fname, car[5][0].item())
    
    image_location =LOCAL_SOURCE + '/{}/{}'.format(test_train.get('imgDir'), car[5][0].item())
    image_tmp = cv2.imread(LOCAL_SOURCE + '/{}/{}'.format(test_train.get('imgDir'), car[5][0].item()))
    car_image[i][:][:][:] = cv2.resize(image_tmp, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC).astype(int)/255
    # car_image = car_image/255
    i = i + 1
    print(i)
  
  # writing into HDF5 files
  with h5py.File('{}/{}'.format(OUTPUT_PATH, test_train.get('fname')), 'w') as hf:
    Xset = hf.create_dataset(name='dataset_x',
      data=car_image,
      shape=(NUM_IMAGES,HEIGHT, WIDTH, CHANNELS),
      maxshape=(NUM_IMAGES,HEIGHT, WIDTH, CHANNELS),
      dtype = np.uint8,
      compression="gzip",
      compression_opts=9)
    yset = hf.create_dataset(name='dataset_y',
      data = (car_class),
      shape=(NUM_IMAGES,),
      maxshape=(NUM_IMAGES,),
      dtype = np.uint8,
      compression="gzip",
      compression_opts=9)
    