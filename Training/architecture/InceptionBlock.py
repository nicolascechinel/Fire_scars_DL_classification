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
from keras.layers import Layer

class InceptionBlock:
    def __init__(self, output_channels, conv1x1_filters, conv3x3_reduce_filters, conv3x3_filters, conv5x5_reduce_filters, conv5x5_filters, pool_filters):
        self.output_channels = output_channels
        self.conv1x1_filters = conv1x1_filters
        self.conv3x3_reduce_filters = conv3x3_reduce_filters
        self.conv3x3_filters = conv3x3_filters
        self.conv5x5_reduce_filters = conv5x5_reduce_filters
        self.conv5x5_filters = conv5x5_filters
        self.pool_filters = pool_filters

    def __call__(self, x):
        conv1x1 = Conv2D(self.conv1x1_filters, (1, 1), padding="same", activation="swish")(x)
        
        conv3x3_reduce = Conv2D(self.conv3x3_reduce_filters, (1, 1), padding="same", activation="swish")(x)
        conv3x3 = Conv2D(self.conv3x3_filters, (3, 3), padding="same", activation="swish")(conv3x3_reduce)
        
        conv5x5_reduce = Conv2D(self.conv5x5_reduce_filters, (1, 1), padding="same", activation="swish")(x)
        conv5x5 = Conv2D(self.conv5x5_filters, (5, 5), padding="same", activation="swish")(conv5x5_reduce)
        
        pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        pool_proj = Conv2D(self.pool_filters, (1, 1), padding="same", activation="swish")(pool)
        
        output = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, pool_proj])
        
        return output