# This contains the architecture for the VGG family of networks
# These networks were introduced in the following paper :
# https://arxiv.org/abs/1409.1556v6


import tensorflow as tf

import utils


def vgg16(
        image_height,
        image_width,
        num_classes=None,
        model_name='vgg16',
        data_format='channels_first',
):
    """
    Constructs a VGG16 model architecture.
    The model is described as VGG-D in the original paper
    :param image_height: Image height. Can be None.
    :param image_width: Image width. Can be None.
    :param num_classes: Number of classes. If None, FC layers are not constructed.
    :param model_name: Name of the model.
    :param data_format: One of 'channels_first' and 'channels_last'.
    :return:
    """
    if data_format == 'channels_first':
        input_shape = (3, image_height, image_width)
    else:
        input_shape = (image_height, image_width, 3)

    x = tf.keras.layers.Input(
        shape=input_shape,
        name='Input'
    )

    input_tensor = x
    x = vgg_conv_output(x, data_format=data_format,
                        vgg_configuration=[
                            [64] * 2,
                            [128] * 2,
                            [256] * 3,
                            [512] * 3,
                            [512] * 3
                        ]
                        )

    if num_classes is not None:
        x = tf.keras.layers.Flatten(
            data_format=data_format, name='flattened')(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
            kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001),
            name='fc1'
        )(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
            kernel_regularizer=tf.keras.regularizers.L2(l2=0.0001),
            name='fc2'
        )(x)

        x = tf.keras.layers.Dense(
            units=num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=0),
            name='logits'
        )(x)

    model = tf.keras.Model(inputs=[input_tensor], outputs=[x], name=model_name)
    return model


def vgg_conv_output(x,
                    data_format,
                    vgg_configuration):
    for block_num, block_conf in enumerate(vgg_configuration):
        for layer_num, filter_num in enumerate(block_conf):
            x = utils.conv_block(
                x,
                num_filters=filter_num,
                kernel_height=3,
                kernel_width=3,
                stride_height=1,
                stride_width=1,
                activation='relu',
                padding='same',
                data_format=data_format,
                use_bias=True,
                base_name='block-{}/{}'.format(block_num + 1, layer_num + 1)
            )

        x = tf.keras.layers.MaxPool2D(
            pool_size=2,
            strides=2,
            data_format=data_format,
            name='pool-{}'.format(block_num + 1)
        )(x)

    return x


def vgg19(
        image_height,
        image_width,
        num_classes=None,
        model_name='vgg19',
        data_format='channels_first',
):
    """
    Constructs a VGG19 model architecture.
    The model is described as VGG-E in the original paper
    :param image_height: Image height. Can be None.
    :param image_width: Image width. Can be None.
    :param num_classes: Number of classes. If None, FC layers are not constructed.
    :param model_name: Name of the model.
    :param data_format: One of 'channels_first' and 'channels_last'.
    :return:
    """
    if data_format == 'channels_first':
        input_shape = (3, image_height, image_width)
    else:
        input_shape = (image_height, image_width, 3)

    x = tf.keras.layers.Input(
        shape=input_shape,
        name='Input'
    )

    input_tensor = x
    x = vgg_conv_output(x, data_format=data_format,
                        vgg_configuration=[
                            [64] * 2,
                            [128] * 2,
                            [256] * 4,
                            [512] * 4,
                            [512] * 4
                        ]
                        )

    if num_classes is not None:
        x = tf.keras.layers.Flatten(
            data_format=data_format, name='flattened')(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            name='fc1'
        )(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            name='fc2'
        )(x)

        x = tf.keras.layers.Dense(
            units=num_classes,
            activation=None,
            use_bias=True,
            name='logits'
        )(x)

    model = tf.keras.Model(inputs=[input_tensor], outputs=[x], name=model_name)
    return model


def vgg13(
        image_height,
        image_width,
        num_classes=None,
        model_name='vgg13',
        data_format='channels_first',
):
    """
    Constructs a VGG13 model architecture.
    The model is described as VGG-B in the original paper
    :param image_height: Image height. Can be None.
    :param image_width: Image width. Can be None.
    :param num_classes: Number of classes. If None, FC layers are not constructed.
    :param model_name: Name of the model.
    :param data_format: One of 'channels_first' and 'channels_last'.
    :return:
    """
    if data_format == 'channels_first':
        input_shape = (3, image_height, image_width)
    else:
        input_shape = (image_height, image_width, 3)

    x = tf.keras.layers.Input(
        shape=input_shape,
        name='Input'
    )

    input_tensor = x
    x = vgg_conv_output(x, data_format=data_format,
                        vgg_configuration=[
                            [64] * 2,
                            [128] * 2,
                            [256] * 2,
                            [512] * 2,
                            [512] * 2
                        ]
                        )

    if num_classes is not None:
        x = tf.keras.layers.Flatten(
            data_format=data_format, name='flattened')(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            name='fc1'
        )(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            name='fc2'
        )(x)

        x = tf.keras.layers.Dense(
            units=num_classes,
            activation=None,
            use_bias=True,
            name='logits'
        )(x)

    model = tf.keras.Model(inputs=[input_tensor], outputs=[x], name=model_name)
    return model


def vgg11(
        image_height,
        image_width,
        num_classes=None,
        model_name='vgg11',
        data_format='channels_first',
):
    """
    Constructs a VGG11 model architecture.
    The model is described as VGG-A in the original paper
    :param image_height: Image height. Can be None.
    :param image_width: Image width. Can be None.
    :param num_classes: Number of classes. If None, FC layers are not constructed.
    :param model_name: Name of the model.
    :param data_format: One of 'channels_first' and 'channels_last'.
    :return:
    """
    if data_format == 'channels_first':
        input_shape = (3, image_height, image_width)
    else:
        input_shape = (image_height, image_width, 3)

    x = tf.keras.layers.Input(
        shape=input_shape,
        name='Input'
    )

    input_tensor = x
    x = vgg_conv_output(x, data_format=data_format,
                        vgg_configuration=[
                            [64] * 1,
                            [128] * 1,
                            [256] * 2,
                            [512] * 2,
                            [512] * 2
                        ]
                        )

    if num_classes is not None:
        x = tf.keras.layers.Flatten(
            data_format=data_format, name='flattened')(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            name='fc1'
        )(x)

        x = tf.keras.layers.Dense(
            units=4096,
            activation='relu',
            use_bias=True,
            name='fc2'
        )(x)

        x = tf.keras.layers.Dense(
            units=num_classes,
            activation=None,
            use_bias=True,
            name='logits'
        )(x)

    model = tf.keras.Model(inputs=[input_tensor], outputs=[x], name=model_name)
    return model
