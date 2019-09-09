import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weights_path=None):
        inputShape = (width, height, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, width, height)
        
        # initialize the model
        model = Sequential()
        
        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # define the first FC => ACTIVATION layers
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        
        # define the second FC => ACTIVATION layers
        model.add(Dense(84, activation='relu'))
        
        # define the last FC layer (output layer) and define the soft-max classifier
        model.add(Dense(classes, activation='softmax'))

        # if a weights path is supplied (inicating that the model was pre-trained), then load the weights
        if weights_path is not None:
            model.load_weights(weights_path)

        # return the constructed network architecture
        return model
