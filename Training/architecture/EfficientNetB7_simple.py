from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, AveragePooling2D, Dropout, GlobalAveragePooling2D, Dense, Input, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class EfficientNetB7_simple:
    @staticmethod
    def build(width, height, depth, classes):
        # Input
        inputs = Input(shape=(height, width, depth))
        x = inputs

        # Stem
        x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        # Block 1
        x = EfficientNetB7_simple._build_block(x, filters=32, kernel_size=3, strides=1, num_repeat=1, expand_ratio=1)
        x = EfficientNetB7_simple._build_block(x, filters=64, kernel_size=3, strides=2, num_repeat=2, expand_ratio=6)
        x = EfficientNetB7_simple._build_block(x, filters=128, kernel_size=3, strides=2, num_repeat=2, expand_ratio=6)
        x = EfficientNetB7_simple._build_block(x, filters=256, kernel_size=3, strides=2, num_repeat=3, expand_ratio=6)
        x = EfficientNetB7_simple._build_block(x, filters=512, kernel_size=3, strides=2, num_repeat=3, expand_ratio=6)
        x = EfficientNetB7_simple._build_block(x, filters=1024, kernel_size=3, strides=2, num_repeat=1, expand_ratio=6)

        # Head
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(units=classes, activation='softmax')(x)

        # Model
        model = Model(inputs=inputs, outputs=x)
        return model

    @staticmethod
    def _build_block(inputs, filters, kernel_size, strides, num_repeat, expand_ratio):
        x = inputs
        in_channels = K.int_shape(x)[-1]

        # Expand
        expanded_channels = expand_ratio * in_channels
        x = Conv2D(filters=expanded_channels, kernel_size=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        # Depthwise conv
        x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('swish')(x)

        # Project
        x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        # Residual connections
        if strides == 1 and in_channels == filters:
            for i in range(num_repeat):
                x = inputs + x
        return x