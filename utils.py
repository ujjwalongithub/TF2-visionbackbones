import tensorflow as tf
from loguru import logger


def get_channel_axis(data_format):
    """
    Returns the channel axis based on the specified data_format.
    :param data_format: Either "channels_first" or "channels_last"
    :return: 1 ( if channels_first ) else -1
    """
    if data_format == 'channels_first':
        return 1

    return -1


def get_activation(input_batch, activation_name, layer_name):
    """
    Returns the output of an activation layer given an input tensor
    :param input_batch: An input tensor
    :param activation_name: Name of the activation function
    :param layer_name: Name of the layer in a model
    :return: An output tensor after the application of the activation function.
    """
    return tf.keras.layers.Activation(activation=activation_name, name='{}/conv_postact'.format(layer_name))(
        input_batch)



def conv_block(
        x,
        num_filters,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding,
        data_format='channels_first',
        activation=None,
        dilation_height=1,
        dilation_width=1,
        use_bias=False,
        batchnorm_config=None,
        dropout_config=None,
        base_name=None
):
    """
    Constructs a conv block which encapsulates the
    operations of convolution, activction, batch normalization and
    dropout into one coherent function
    :param x: Input tensor
    :param num_filters: Number of convolutional filters
    :param kernel_height: Height of the convolutional filter
    :param kernel_width: Width of the convolutional filter
    :param stride_height: Height of the stride taken by the convolutional filter
    :param stride_width: Width of the stride taken by the convolutional filter
    :param padding: Either "valid" or "same"
    :param data_format: Either "channels_first" or "channels_last"
    :param activation: Name of the activation function
    :param dilation_height: Dilation rate in the vertical direction
    :param dilation_width: Dilation rate in the horizontal direction
    :param use_bias: If true use bias with convolution
    :param batchnorm_config: A dictionary containing the values of the batch
    normalization arguments
    :param dropout_config: A dictionary containing the values of the dropout
    arguments
    :param base_name: Root prefix for the layers in the convolutional block
    :return: An output tensor of the convolutional block
    """
    if base_name is None:
        base_name = 'block'
    x = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=(kernel_height, kernel_width),
        strides=(stride_height, stride_width),
        dilation_rate=(dilation_height, dilation_width),
        data_format=data_format,
        use_bias=use_bias,
        padding=padding,
        activation=None,
        name='{}/conv_preact'.format(base_name)
    )(x)

    if activation is not None:
        x = get_activation(x, activation_name=activation,
                           layer_name='{}/conv_postact'.format(base_name))
    else:
        logger.debug(
            'No activation specified for the conv block {}.'.format(base_name))

    if batchnorm_config is not None:
        x = tf.keras.layers.BatchNormalization(
            axis=get_channel_axis(data_format=data_format),
            **batchnorm_config
        )(x)
    else:
        logger.debug(
            'No batch normalization specified for the conv block {}.'.format(base_name))

    if dropout_config is not None:
        x = tf.keras.layers.Dropout(
            name='{}/dropout'.format(base_name), **dropout_config)(x)
    else:
        logger.debug(
            'No dropout configuration specified for the conv block {}.'.format(base_name))

    return x
