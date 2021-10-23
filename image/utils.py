import typing

import tensorflow as tf


def get_channel_axis(data_format: typing.Union[None, str]) -> int:
    """
    Returns the channel axis corresponding to "data_format".
    This is a useful function to reduce verbosity when writing codes which
    are required to be flexible enough to accept both "channels_first" and
    "channels_last" data formats.
    :param data_format: One of "channels_first", "channels_last" and None.
    If None, uses the output from tf.keras.backend.image_data_format().
    :return: The channel axis corresponding to the image data format
    """
    if data_format is None:
        data_format = tf.keras.backend.image_data_format()

    if data_format == 'channels_first':
        axis = 1
    else:
        axis = -1

    return axis


def get_activation(input_batch, activation_name, layer_name):
    """
    Returns the output of an activation layer given an input tensor.
    Currently, this function only supports activation function names
    inbuilt in TensorFlow. This behavior may change in the future as
    more activation functions unavailable in native TF API are implemented.

    NOTE: The name of this layer is always conv_postact and suffixes the parent
    layer name.
    :param input_batch: An input tensor
    :param activation_name: Name of the activation function
    :param layer_name: Name of the layer in a model
    :return: An output tensor after the application of the activation function.
    """
    return tf.keras.layers.Activation(activation=activation_name,
                                      name='{}/conv_postact'.format(
                                          layer_name))(
        input_batch)


class ZeroPadding2D(tf.keras.layers.Layer):
    """
    This layer performs zero padding for image-based tensors.
    """
    #TODO: Make the documentation more extensive
    def __init__(self,
                 padding_x=None,
                 padding_y=None,
                 data_format: typing.Union[None, str] = None,
                 layer_name: typing.Union[None, str] = None
                 ):
        super(ZeroPadding2D, self).__init__(
            name='{}/padding2d'.format(layer_name)
        )
        if padding_y is None:
            padding_y = [0, 0]
        if padding_x is None:
            padding_x = [0, 0]
        if data_format is None:
            data_format = tf.keras.backend.image_data_format()
        if data_format not in ['channels_first', 'channels_last']:
            raise ValueError(
                'data_format must be one of "channels_first" and "channels_last".')
        if data_format == 'channels_first':
            paddings = [
                [0, 0],
                [0, 0],
                padding_y,
                padding_x
            ]
        else:
            paddings = [
                [0, 0],
                padding_y,
                padding_x,
                [0, 0]
            ]
        self._paddings = paddings

    def call(self, x):
        x = tf.pad(
            x,
            paddings=self._paddings
        )
        return x
