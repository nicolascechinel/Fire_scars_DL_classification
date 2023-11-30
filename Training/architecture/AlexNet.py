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

class AlexNet:
	@staticmethod
	def build(width, height, depth, classes):

		#RESAMPLING
		#Input = 227x227x3

		# initialize the model 
		model = Sequential()
		inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# first set of CONV => RELU => POOL layers
		#model.add(Conv2D(96, (11, 11),strides=(4, 4), activation="relu", input_shape=inputShape))
		model.add(Conv2D(96, (3, 3),strides=(1, 1), activation="relu", padding="same", input_shape=inputShape))
		#model.add(BatchNormalization())		
		#model.add(MaxPooling2D(pool_size=(3, 3), strides= (2, 2)))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides= (1, 1)))		
		model.add(Dropout(.5))

		# second set of CONV => RELU => POOL layers
		#model.add(Conv2D(256, (5, 5), strides=(1, 1), activation="relu", padding="same"))
		model.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
		#model.add(BatchNormalization())
		#model.add(MaxPooling2D(pool_size=(3, 3), strides= (2, 2)))
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides= (1, 1)))
		model.add(Dropout(.2))			
		
		#Third
		model.add(Conv2D(384, (3, 3), strides=(1, 1), activation="relu", padding="same"))
		#model.add(BatchNormalization())
		model.add(MaxPooling2D(pool_size=(2, 2), padding='same', strides=(1,1)))		
		model.add(Dropout(.2))

		#Fourth
		model.add(Conv2D(384, (3, 3), strides=(1, 1), activation="relu", padding="same"))
		#model.add(BatchNormalization())
		model.add(AveragePooling2D(pool_size=(2, 2), padding='same', strides=(1,1)))
		model.add(Dropout(.2))

		#Fith
		model.add(Conv2D(256, (3, 3), strides=(1, 1), activation="relu", padding="same"))
		#model.add(BatchNormalization())
		#model.add(MaxPooling2D(pool_size=(3, 3), strides= (2, 2)))
		model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
		model.add(Dropout(.2))	
       
		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(units=4096, activation='relu'))
		model.add(Dropout(.5))
		model.add(Dense(units=4096, activation='relu'))
		model.add(Dropout(.5))
		model.add(Dense(classes, activation="softmax"))		
		return model