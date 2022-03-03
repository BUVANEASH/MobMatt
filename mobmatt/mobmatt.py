import tensorflow as tf
import tensorflow_addons as tfa

def relu6(x):
    """Relu 6."""
    return tf.nn.relu6(x)

def hard_swish(x):
    """Hard swish."""
    return x * tf.nn.relu6(x + 3.0) / 6.0

class ConvBlock(tf.keras.layers.Layer):
    """Convolution Block
    """
    def __init__(self, filters, kernel = (3, 3), strides = 1, act = 'RE', padding = 'same'):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel, padding=padding, strides=strides)
        self.batchnorm = tf.keras.layers.BatchNormalization(axis=-1)
        if act == 'HS':
            self.act = tf.keras.layers.Activation(hard_swish)
        if act == 'RE':
            self.act = tf.keras.layers.Activation(relu6)

    def call(self, x, training = False):
        x = self.conv(x)
        x = self.batchnorm(x, training = training)
        x = self.act(x)
        return x

class Squeeze(tf.keras.layers.Layer):
    """Squeeze and Excitation.
    """

    def __init__(self, filters):
        super(Squeeze, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        self.fullyconnected1 = tf.keras.layers.Dense(self.filters)
        self.fullyconnected2 = tf.keras.layers.Dense(self.input_channels)

    def call(self, inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = self.fullyconnected1(x)
        x = self.fullyconnected2(x)
        x = tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)
        x = tf.math.multiply(inputs, x)
        return x

class Bottleneck(tf.keras.layers.Layer):
    """Bottleneck
    """

    def __init__(self, filters, kernel, expansion, strides, squeeze, act, alpha=1.0):
        super(Bottleneck, self).__init__()
        self.filters = filters
        self.strides = strides
        
        if act == 'HS':
            self.act = tf.keras.layers.Activation(hard_swish)
        if act == 'RE':
            self.act = tf.keras.layers.Activation(relu6)
        
        self.squeeze = squeeze
        if self.squeeze:
            self.squeeze_layer = Squeeze(filters = filters//2)
        
        expand_channels = int(expansion)
        compress_channels = int(alpha * filters)

        self.conv_block = ConvBlock(expand_channels, kernel=(1, 1), strides=(1, 1), act=act)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel, strides=(strides, strides), depth_multiplier=1, padding='same')
        self.batchnorm1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.conv2d = tf.keras.layers.Conv2D(compress_channels, (1, 1), strides=(1, 1), padding='same')
        self.batchnorm2 = tf.keras.layers.BatchNormalization(axis=-1)

    def build(self, input_shape):
        self.residual_connection = self.strides == 1 and input_shape[3] == self.filters

    def call(self, inputs, training = False):
        x = self.conv_block(inputs)
        x = self.depthwise_conv(x)
        x = self.batchnorm1(x, training = training)
        x = self.act(x)
        if self.squeeze:
            x = self.squeeze_layer(x)
        x = self.conv2d(x)
        x = self.batchnorm2(x, training = training)
        if self.residual_connection:
            x = x + inputs
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        """Decoder
        """
        super(Decoder, self).__init__()
        self.bottleneck1 = Bottleneck(48, (5, 5), 1152, 1, True, act = 'HS', alpha=1.0)
        self.bottleneck2 = Bottleneck(24, (5, 5), 192, 1, True, act = 'HS', alpha=1.0)
        self.bottleneck3 = Bottleneck(16, (3, 3), 96, 1, False, act = 'RE', alpha=1.0)
        self.bottleneck4 = Bottleneck(16, (3, 3), 64, 1, False, act = 'RE', alpha=1.0)
        self.bottleneck5 = Bottleneck(16, (3, 3), 64, 1, True, act = 'RE', alpha=1.0)
        self.act = tf.keras.layers.Activation(relu6)
        self.conv2d = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')
        self.linear = tf.keras.layers.Activation('linear', dtype = tf.float32)
    
    def call(self, inputs, training = False):
        feat1, feat2, feat3, feat4, feat5 = inputs

        x = self.bottleneck1(feat5, training = training)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation = 'bilinear')(x)

        x = tf.cast(x, dtype = feat5.dtype)
        x = tf.concat([x,feat4], axis = -1)
        x = self.bottleneck2(x, training = training)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation = 'bilinear')(x)

        x = tf.cast(x, dtype = feat4.dtype)        
        x = tf.concat([x,feat3], axis = -1)
        x = self.bottleneck3(x, training = training)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation = 'bilinear')(x)

        x = tf.cast(x, dtype = feat3.dtype)        
        x = tf.concat([x,feat2], axis = -1)
        x = self.bottleneck4(x, training = training)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation = 'bilinear')(x)

        x = tf.cast(x, dtype = feat2.dtype)        
        x = tf.concat([x,feat1], axis = -1)
        x = self.bottleneck5(x, training = training)
        x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation = 'bilinear')(x)
        x = tf.cast(x, dtype = feat1.dtype)
        
        x  = self.act(x)
        x = self.conv2d(x)
        x = self.linear(x)
        
        return x

class MobMatt(tf.keras.Model):
    """
    MobileNetV3 Portrait Matting Model
    """
    def __init__(self, image_size, freeze_encoder = True, name = "MobMatt", **kwargs):
        """
        Model Subclas init

        Args:
            image_size (int): Input image size
        """
        super(MobMatt, self).__init__(name = name, **kwargs)
        """
        |=====================================================|
        | Layer Names                         | Layer Outputs |
        |=====================================|===============|
        | multiply                            | 112*112*16    |
        | expanded_conv/project/BatchNorm     | 56*56*16      |
        | expanded_conv_2/Add                 | 28*28*24      |
        | expanded_conv_6/project/BatchNorm   | 14*14*48      |
        | multiply_17                         | 7*7*576       |
        |=====================================================|
        """
        self.freeze_encoder = freeze_encoder
        self.image_size = image_size
        self.backbone_layers = ['multiply','expanded_conv/project/BatchNorm','expanded_conv_2/Add','expanded_conv_6/project/BatchNorm','multiply_17']
        self.backbone = tf.keras.applications.MobileNetV3Small(input_shape=(self.image_size,self.image_size,3), include_top=False)

        self.backbone_out_list = []
        for l in self.backbone.layers:
            if l.name in self.backbone_layers:
                self.backbone_out_list.append(l.output)

        self.encoder = tf.keras.Model(inputs = self.backbone.input, outputs = self.backbone_out_list)
        
        if self.freeze_encoder:
            for l in self.encoder.layers:
                if l.trainable:
                    l.trainable = False

        self.decoder = Decoder()

    def call(self, inputs, training = False):
        x = inputs
        x = self.encoder(x, training = training)
        x = self.decoder(x, training = training)
        return x
    
    def model(self):
        y = tf.keras.layers.Input(shape=(self.image_size, self.image_size, 3), name = "Input")
        return tf.keras.Model(inputs=y, outputs=[self.call(y)], name = self.name)