"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.
Description :
Author：Team Li
"""

import sys
sys.path.append('..')

import time
import random
import functools
import numpy as np
from models.example_model import Model,Transform
from utils.utils import get_img
import tensorflow as tf

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
DEVICES = 'CUDA_VISIBLE_DEVICES'

class Trainer:
    def __init__(self, content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000, logger = None,
             batch_size=4, save_path='saver/fns.ckpt', learning_rate=1e-3):
        self.content_targets = content_targets
        self.style_target = style_target
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.vgg_path = vgg_path
        self.epochs = epochs
        self.print_iterations = print_iterations
        self.batch_size = batch_size
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.logger = logger

        self.model = Model()
        self.transform = Transform()

    def train(self):
        mod = len(self.content_targets) % self.batch_size
        if mod > 0:
            print("Train set has been trimmed slightly..")
            self.content_targets = self.content_targets[:-mod]

        style_features = {}
        batch_shape = (self.batch_size, 256, 256, 3)
        style_shape = (1,) + self.style_target.shape

        # precompute style features
        with tf.Graph().as_default(), tf.device('/cpu:0'), tf.compat.v1.Session() as sess:
            style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
            style_image_pre = self.model.preprocess(style_image)
            net = self.model.build_model(self.vgg_path, style_image_pre)
            style_pre = np.array([self.style_target])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={style_image:style_pre})
                features = np.reshape(features, (-1, features.shape[3]))
                gramMatrix = np.matmul(features.T, features) / features.size
                style_features[layer] = gramMatrix

        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            X_content = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name="X_content")

            # precompute content features
            X_pre = self.model.preprocess(X_content)
            content_features = {}
            content_net = self.model.build_model(self.vgg_path, X_pre)
            content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

            # conten loss
            preds = self.transform.build_model(X_content / 255.0)
            preds_pre = self.model.preprocess(preds)
            net = self.model.build_model(self.vgg_path, preds_pre)
            content_size = self._tensor_size(content_features[CONTENT_LAYER]) * self.batch_size
            assert self._tensor_size(content_features[CONTENT_LAYER]) == self._tensor_size(net[CONTENT_LAYER])
            content_loss = self.content_weight * (2 * tf.nn.l2_loss(
                net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size)

            # style loss
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                bs, height, width, filters = map(lambda i: i, layer.get_shape())
                size = height * width * filters
                feats = tf.reshape(layer, (bs, height * width, filters))
                feats_T = tf.transpose(a=feats, perm=[0, 2, 1])
                grams = tf.matmul(feats_T, feats) / size
                style_gram = style_features[style_layer]
                style_losses.append(2 * tf.nn.l2_loss(grams - style_gram) / style_gram.size)
            style_loss = self.style_weight * functools.reduce(tf.add, style_losses) / self.batch_size

            # total variation denoising
            tv_y_size = self._tensor_size(preds[:, 1:, :, :])
            tv_x_size = self._tensor_size(preds[:, :, 1:, :])
            y_tv = tf.nn.l2_loss(preds[:, 1:, :, :] - preds[:, :batch_shape[1] - 1, :, :])
            x_tv = tf.nn.l2_loss(preds[:, :, 1:, :] - preds[:, :, :batch_shape[2] - 1, :])
            tv_loss = self.tv_weight * 2 * (x_tv / tv_x_size + y_tv / tv_y_size) / self.batch_size

            # overall loss
            loss = content_loss + style_loss + tv_loss
            train_step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(loss)
            sess.run(tf.compat.v1.global_variables_initializer())
            uid = random.randint(1, 100)
            if self.logger:
                self.logger.info("UID: %s" % uid)
            for epoch in range(self.epochs):
                start_time = time.time()
                num_examples = len(self.content_targets)
                iterations = 0
                while iterations * self.batch_size < num_examples:
                    curr = iterations * self.batch_size
                    step = curr + self.batch_size
                    X_batch = np.zeros(batch_shape, dtype=np.float32)
                    for j, img_p in enumerate(self.content_targets[curr:step]):
                        X_batch[j] = get_img(img_p, (256, 256, 3)).astype(np.float32)

                    iterations += 1
                    assert X_batch.shape[0] == self.batch_size, 'X_batch.shape[0] != self.batch_size'
                    train_step.run(feed_dict={X_content: X_batch})
                    is_print_iter = int(iterations) % self.print_iterations == 0
                    is_last = epoch == self.epochs - 1 and iterations * self.batch_size >= num_examples
                    should_print = is_print_iter or is_last
                    if should_print:
                        to_get = [style_loss, content_loss, tv_loss, loss, preds]

                        tup = sess.run(to_get, feed_dict={X_content: X_batch})
                        _style_loss, _content_loss, _tv_loss, _loss, _preds = tup
                        losses = (_style_loss, _content_loss, _tv_loss, _loss)
                        saver = tf.compat.v1.train.Saver()
                        saver.save(sess, self.save_path)
                        yield (_preds, losses, iterations, epoch)
                end_time = time.time()
                delta_time = end_time - start_time
                if self.logger:
                    self.logger.info("UID: %s, Epoch time: %s" % (uid, delta_time))

    def _tensor_size(self, tensor):
        from operator import mul
        return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
