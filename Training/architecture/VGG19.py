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

class VGG19:
	@staticmethod
	def build(width, height, depth, classes):       
		# initialize the model		
		model = Sequential()
		inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)		
		# Block 1
		model.add(Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=inputShape))		
		model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))		
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
		#model.add(Dropout(.25))

		# Block 2
		model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))		
		model.add(Conv2D(128, (3, 3), padding="same", activation='relu'))		
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
		#model.add(Dropout(.25))

		# Block 3
		model.add(Conv2D(256, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(256, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(256, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(256, (3, 3), padding= 'same', activation='relu'))		
		model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same', data_format=None))
		#model.add(MaxPooling2D(padding='same'))
		#model.add(Dropout(.25))

		# Block 4
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))				
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))		
		model.add(MaxPooling2D(padding='same'))
		#model.add(Dropout(.25))
		
		# Block 5
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))	
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))		
		model.add(Conv2D(512, (3, 3), padding= 'same', activation='relu'))		
		model.add(MaxPooling2D(padding='same'))
		#model.add(Dropout(.25))

		# Fully connected layers
		model.add(Flatten())
		model.add(Dense(4096, activation= 'relu'))
		model.add(Dropout(.5))
		model.add(Dense(4096, activation= 'relu'))
		model.add(Dropout(.5))		
		# Softmax classifier
		model.add(Dense(classes, activation='softmax'))		
		return model