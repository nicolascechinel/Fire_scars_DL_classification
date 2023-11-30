from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization

class LeNet:
	@staticmethod
	def build(width, height, depth, classes):        
		# initialize the model		
		model = Sequential()
		inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(100, (3, 3), padding="same", input_shape=inputShape))
		#model.add(BatchNormalization())
		model.add(Activation("relu"))
		#model.add(AveragePooling2D())
		model.add(MaxPooling2D(pool_size=(2, 2), padding= 'same'))	
		model.add(Dropout(.25))
		
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(150, (3, 3), padding= 'same'))
		#model.add(BatchNormalization())
		model.add(Activation("relu"))
		#model.add(AveragePooling2D())
		model.add(MaxPooling2D((2, 2), strides=(1, 1)))
		model.add(Dropout(.25))

		# Third set of CONV => RELU => POOL layers
		model.add(Conv2D(300, (3, 3), padding="same"))
		#model.add(BatchNormalization())
		model.add(Activation("relu"))
		#model.add(AveragePooling2D())
		model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
		#model.add(Dropout(.25))
		
		# Four set of CONV => RELU => POOL layers
		model.add(Conv2D(530, (3, 3), padding="same"))
		#model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(AveragePooling2D(padding='same'))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
		model.add(Dropout(.25))
	      
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(units=1000))
		#model.add(BatchNormalization())
		model.add(Activation("relu"))	
		model.add(Dropout(.4))
		model.add(Dense(units=700))
		#model.add(BatchNormalization())
		model.add(Activation("relu"))
		model.add(Dropout(.4))
		# softmax classifier
		model.add(Dense(classes))
		#model.add(BatchNormalization())
		model.add(Activation("softmax"))
		return model

"""
		#initialize the model
		model = Sequential()
		inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
		# first set of CONV => RELU => POOL layers
		model.add(Conv2D(6, (3, 3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		##model.add(AveragePooling2D())
		model.add(MaxPooling2D((2,2), padding='same'))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))	
		model.add(Dropout(.25))
		# second set of CONV => RELU => POOL layers
		model.add(Conv2D(16, (3, 3), padding="same"))
		model.add(Activation("relu"))
		##model.add(AveragePooling2D())
		model.add(MaxPooling2D((2,2), padding='same'))
		#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(.25))
		# first (and only) set of FC => RELU layers
		# Third set of CONV => RELU => POOL layers
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		#model.add(AveragePooling2D())
		model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		# Four set of CONV => RELU => POOL layers
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(AveragePooling2D(padding='same'))
		# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Dropout(.25))
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(units=120, activation='relu'))
		model.add(Dropout(.4))
		model.add(Dense(units= 84, activation='relu'))
		model.add(Dropout(.4))
		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		# return the constructed network architecture
		return model
"""
