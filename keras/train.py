# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os
from tqdm import tqdm
from lenet import LeNet

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-test", "--data_test", required = True, help = "path to input data_test")
    ap.add_argument("-train", "--data_train", required = True, help = "path to input data_train")
    ap.add_argument("-m", "--model", required = True, help = "path to output model")
    args = vars(ap.parse_args()) 
    return args


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 5
INIT_LR = 1e-3
BS = 64
CLASS_NUM = 10
NORM_SIZE = 64

def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in tqdm(imagePaths):
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        if image is None:
            print (imagePath)
            continue
        image = cv2.resize(image, (NORM_SIZE, NORM_SIZE))
        image = img_to_array(image)
        data.append(image)
        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])       
        labels.append(label)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype = "float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes = CLASS_NUM)                         
    return data, labels


def train(aug, trainX, trainY, testX, testY, args):
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width = NORM_SIZE, height = NORM_SIZE, depth = 3, classes = CLASS_NUM)
    opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size = BS), validation_data = (testX, testY), steps_per_epoch = len(trainX) // BS, epochs = EPOCHS, verbose = 1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    

if __name__=='__main__':
    args = args_parse()
    train_file_path = args["data_train"]
    test_file_path = args["data_test"]
    trainX,trainY = load_data(train_file_path)
    testX,testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range = 30, width_shift_range = 0.1, height_shift_range = 0.1, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = False, fill_mode = "nearest")
    train(aug, trainX, trainY, testX, testY, args)