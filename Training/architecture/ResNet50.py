from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.regularizers import l2

class ResNet50:
	@staticmethod
	def build(width, height, depth, classes):
		input_shape = (height, width, depth)
		inputs = Input(shape=input_shape)
		x = ZeroPadding2D(padding=(3, 3))(inputs)        
		# Stage 1
		x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = ZeroPadding2D(padding=(1, 1))(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
		# Stage 2
		x = ResNet50.conv_block(x, filters=[64, 64, 256], strides=(1, 1))
		x = ResNet50.identity_block(x, filters=[64, 64, 256])
		x = ResNet50.identity_block(x, filters=[64, 64, 256])        
		# Stage 3
		x = ResNet50.conv_block(x, filters=[128, 128, 512], strides=(2, 2))
		x = ResNet50.identity_block(x, filters=[128, 128, 512])
		x = ResNet50.identity_block(x, filters=[128, 128, 512])        
		x = ResNet50.identity_block(x, filters=[128, 128, 512])        
		# Stage 4
		x = ResNet50.conv_block(x, filters=[256, 256, 1024], strides=(2, 2))
		x = ResNet50.identity_block(x, filters=[256, 256, 1024])
		x = ResNet50.identity_block(x, filters=[256, 256, 1024])        
		x = ResNet50.identity_block(x, filters=[256, 256, 1024])        
		x = ResNet50.identity_block(x, filters=[256, 256, 1024])        
		x = ResNet50.identity_block(x, filters=[256, 256, 1024])        
		# Stage 5
		x = ResNet50.conv_block(x, filters=[512, 512, 2048], strides=(2, 2))
		x = ResNet50.identity_block(x, filters=[512, 512, 2048])
		x = ResNet50.identity_block(x, filters=[512, 512, 2048])
		# Output stage
		x = AveragePooling2D(pool_size=(1, 1))(x)		
		x = Flatten()(x)
		outputs = Dense(classes, activation='softmax')(x)
		# Create model
		model = Model(inputs=inputs, outputs=outputs)
		return model

	@staticmethod
	def identity_block(x, filters):
		filter1, filter2, filter3 = filters
		shortcut = x
		x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x =BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv2D(filters=filter2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		x = Add()([x, shortcut])
		x = Activation('relu')(x)
		return x
	@staticmethod
	def conv_block(x, filters, strides):
		filter1, filter2, filter3 = filters
		shortcut = x
		x = Conv2D(filters=filter1, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv2D(filters=filter2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
		x = BatchNormalization()(x)
		shortcut = Conv2D(filters=filter3, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(shortcut)
		shortcut = BatchNormalization()(shortcut)
		x = Add()([x, shortcut])
		x = Activation('relu')(x)
		return x