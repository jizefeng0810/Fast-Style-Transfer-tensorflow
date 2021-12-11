"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
Author：Team Li
"""

import tensorflow as tf
import numpy as np
import scipy.io

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])
layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

class Model():
    def __init__(self):
        super(Model, self).__init__()

    def build_model(self, data_path, input_image):
        data = scipy.io.loadmat(data_path)
        mean = data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))
        weights = data['layers'][0]

        net = {}
        current = input_image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = self._conv_layer(current, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(current)
            elif kind == 'pool':
                current = self._pool_layer(current)
            net[name] = current

        assert len(net) == len(layers)
        return net

    def _conv_layer(self, input, weights, bias):
        conv = tf.nn.conv2d(input=input, filters=tf.constant(weights), strides=(1, 1, 1, 1),
                            padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def _pool_layer(self, input):
        return tf.nn.max_pool2d(input=input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                                padding='SAME')

    def preprocess(self, image):
        return image - MEAN_PIXEL

    def unprocess(self, image):
        return image + MEAN_PIXEL

class Transform():
    def __init__(self):
        super(Transform, self).__init__()
        self.WEIGHTS_INIT_STDEV = .1

    def build_model(self, image):
        conv1 = self._conv_layer(image, 32, 9, 1)
        conv2 = self._conv_layer(conv1, 64, 3, 2)
        conv3 = self._conv_layer(conv2, 128, 3, 2)
        resid1 = self._residual_block(conv3, 3)
        resid2 = self._residual_block(resid1, 3)
        resid3 = self._residual_block(resid2, 3)
        resid4 = self._residual_block(resid3, 3)
        resid5 = self._residual_block(resid4, 3)
        conv_t1 = self._conv_tranpose_layer(resid5, 64, 3, 2)
        conv_t2 = self._conv_tranpose_layer(conv_t1, 32, 3, 2)
        conv_t3 = self._conv_layer(conv_t2, 3, 9, 1, relu=False)
        preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
        return preds

    def _conv_init_vars(self, net, out_channels, filter_size, transpose=False):
        _, rows, cols, in_channels = [i for i in net.get_shape()]
        if not transpose:
            weights_shape = [filter_size, filter_size, in_channels, out_channels]
        else:
            weights_shape = [filter_size, filter_size, out_channels, in_channels]

        weights_init = tf.Variable(tf.random.truncated_normal(weights_shape, stddev=self.WEIGHTS_INIT_STDEV, seed=1),
                                   dtype=tf.float32)
        return weights_init

    def _conv_layer(self, net, num_filters, filter_size, strides, relu=True):
        weights_init = self._conv_init_vars(net, num_filters, filter_size)
        strides_shape = [1, strides, strides, 1]
        net = tf.nn.conv2d(input=net, filters=weights_init, strides=strides_shape, padding='SAME')
        net = self._instance_norm(net)
        if relu:
            net = tf.nn.relu(net)
        return net

    def _residual_block(self, net, filter_size=3):
        tmp = self._conv_layer(net, 128, filter_size, 1)
        return net + self._conv_layer(tmp, 128, filter_size, 1, relu=False)

    def _instance_norm(self, net, train=True):
        batch, rows, cols, channels = [i for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(x=net, axes=[1, 2], keepdims=True)
        shift = tf.Variable(tf.zeros(var_shape))
        scale = tf.Variable(tf.ones(var_shape))
        epsilon = 1e-3
        normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
        return scale * normalized + shift

    def _conv_tranpose_layer(self, net, num_filters, filter_size, strides):
        weights_init = self._conv_init_vars(net, num_filters, filter_size, transpose=True)

        batch_size, rows, cols, in_channels = [i for i in net.get_shape()]
        new_rows, new_cols = int(rows * strides), int(cols * strides)
        # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

        new_shape = [batch_size, new_rows, new_cols, num_filters]
        tf_shape = tf.stack(new_shape)
        strides_shape = [1, strides, strides, 1]

        net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
        net = self._instance_norm(net)
        return tf.nn.relu(net)