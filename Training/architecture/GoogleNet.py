from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

class GoogleNet:
	@staticmethod
	def conv_module(x, filters, kernel_size, strides, chanDim, padding='same'):
		x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = Activation('relu')(x)
		return x
    
	@staticmethod
	def inception_module(x, filters1x1, filters3x3Reduce, filters3x3, filters5x5Reduce, filters5x5, chanDim):
		conv_1x1 = GoogleNet.conv_module(x, filters1x1, (1, 1), (1, 1), chanDim)
		conv_3x3 = GoogleNet.conv_module(x, filters3x3Reduce, (1, 1), (1, 1), chanDim)
		conv_3x3 = GoogleNet.conv_module(conv_3x3, filters3x3, (3, 3), (1, 1), chanDim)
		conv_5x5 = GoogleNet.conv_module(x, filters5x5Reduce, (1, 1), (1, 1), chanDim)
		conv_5x5 = GoogleNet.conv_module(conv_5x5, filters5x5, (5, 5), (1, 1), chanDim)
		pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
		pool_proj = GoogleNet.conv_module(pool_proj, filters1x1, (1, 1), (1, 1), chanDim)
		x = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=chanDim)
		return x

	@staticmethod
	def build(width, height, depth, classes):
		inputShape = (height, width, depth)
		chanDim = -1
		if K.image_data_format() == 'channels_first':
			inputShape = (depth, height, width)
			chanDim = 1
		inputs = Input(shape=inputShape)
		x = GoogleNet.conv_module(inputs, 64, (5, 5), (1, 1), chanDim)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
		x = GoogleNet.conv_module(x, 64, (1, 1), (1, 1), chanDim)
		x = GoogleNet.conv_module(x, 192, (3, 3), (1, 1), chanDim)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
		x = GoogleNet.inception_module(x, 128, 128, 192, 32, 96, chanDim)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
		x = GoogleNet.inception_module(x, 192, 96, 208, 16, 48, chanDim)
		x = GoogleNet.inception_module(x, 160, 112, 224, 24, 64, chanDim)
		x = GoogleNet.inception_module(x, 128, 128, 256, 24, 64, chanDim)
		x = GoogleNet.inception_module(x, 112, 144, 288, 32, 64, chanDim)
		x = GoogleNet.inception_module(x, 256, 160, 320, 32, 128, chanDim)
		x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
		x = GoogleNet.inception_module(x, 256, 160, 320, 32, 128, chanDim)
		x = GoogleNet.inception_module(x, 384, 192, 384, 48, 128, chanDim)
		x = AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None)(x)
		x = Dropout(0.4)(x)
		x = Flatten()(x)
		x = Dense(classes)(x)
		x = Activation('softmax')(x)
		model = Model(inputs, x, name='googlenet')
		return model