from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class DenseNet121:
	@staticmethod
	def build(width, height, depth, classes):
		input_shape = (height, width, depth)
		# input tensor
		inputs = Input(shape=input_shape)
		# initial convolution layer
		x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

		# dense block 1
		x, nb_filters = DenseNet121.dense_block(x, nb_layers=6, nb_filters=32, growth_rate=32)

		# transition layer 1
		x = DenseNet121.transition_layer(x, nb_filters=nb_filters)

        # dense block 2
		x, nb_filters = DenseNet121.dense_block(x, nb_layers=12, nb_filters=32, growth_rate=32)

        # transition layer 2
		x = DenseNet121.transition_layer(x, nb_filters=nb_filters)

        # dense block 3
		x, nb_filters = DenseNet121.dense_block(x, nb_layers=24, nb_filters=32, growth_rate=32)

        # transition layer 3
		x = DenseNet121.transition_layer(x, nb_filters=nb_filters)

        # dense block 4
		x, nb_filters = DenseNet121.dense_block(x, nb_layers=16, nb_filters=32, growth_rate=32)

        # global average pooling and output
		x = GlobalAveragePooling2D()(x)
		x = Dense(classes, activation='softmax')(x)

        # create model
		model = Model(inputs=inputs, outputs=x)

		return model

	@staticmethod
	def dense_block(x, nb_layers, nb_filters, growth_rate):
		for i in range(nb_layers):
			cb = DenseNet121.conv_block(x, nb_filters=growth_rate)
			x = Concatenate()([x, cb])
			nb_filters += growth_rate
		return x, nb_filters

	@staticmethod
	def conv_block(x, nb_filters):
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Conv2D(filters=nb_filters, kernel_size=(3, 3), padding='same')(x)
		return x

	@staticmethod
	def transition_layer(x, nb_filters):
		x = BatchNormalization()(x)
		x = ReLU()(x)
		x = Conv2D(filters=nb_filters, kernel_size=(1, 1), padding='same')(x)
		x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		return x