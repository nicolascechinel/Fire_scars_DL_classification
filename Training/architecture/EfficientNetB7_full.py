from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from Training.architecture.InceptionBlock import InceptionBlock

class EfficientNetB7_full:
    @staticmethod
    def build(width, height, depth, classes):
        # Input = 600x600x3
        
        # initialize the model 
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # First Convolution layer and Max Pooling layer
        model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", activation="swish", input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # Second Convolution layer
        model.add(Conv2D(64, (1, 1), strides=(1, 1), padding="same", activation="swish"))
        model.add(Conv2D(192, (3, 3), strides=(1, 1), padding="same", activation="swish"))

        # Max Pooling layer
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # Inception Block 1
        model.add(InceptionBlock(192, 64, 96, 128, 16, 32, 32))

        # Inception Block 2
        model.add(InceptionBlock(256, 128, 128, 192, 32, 96, 64))

        # Max Pooling layer
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # Inception Block 3
        model.add(InceptionBlock(480, 192, 96, 208, 16, 48, 64))
        model.add(InceptionBlock(512, 160, 112, 224, 24, 64, 64))
        model.add(InceptionBlock(512, 128, 128, 256, 24, 64, 64))
        model.add(InceptionBlock(512, 112, 144, 288, 32, 64, 64))
        model.add(InceptionBlock(528, 256, 160, 320, 32, 128, 128))

        # Max Pooling layer
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

        # Inception Block 4
        model.add(InceptionBlock(832, 256, 160, 320, 32, 128, 128))
        model.add(InceptionBlock(832, 384, 192, 384, 48, 128, 128))

        # Global Average Pooling layer
        model.add(GlobalAveragePooling2D())

        # Dropout layer
        model.add(Dropout(0.5))

        # Fully Connected layer
        model.add(Dense(classes, activation="softmax"))
        return model
