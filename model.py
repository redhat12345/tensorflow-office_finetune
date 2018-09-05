"""
Derived from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""
import tensorflow as tf
import numpy as np
import math


class AlexNetModel(object):

    def __init__(self, num_classes=1000, dropout_keep_prob=0.5):
        self.num_classes = num_classes
        self.dropout_keep_prob = dropout_keep_prob
        self.featurelen = 256

    def inference(self, x, training=False):
        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(x, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-5, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-5, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
        conv4_flattened = tf.contrib.layers.flatten(conv4)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        self.flattened = flattened
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        if training:
            fc6 = dropout(fc6, self.dropout_keep_prob)
        self.fc6 = fc6
        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        if training:
            fc7 = dropout(fc7, self.dropout_keep_prob)
        self.fc7 = fc7
        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = fc(fc7, 4096, 256, relu=False, name='fc8')
        self.fc8 = fc8
        self.score = fc(fc8, 256, self.num_classes, relu=False, stddev=0.005, name='fc9')
        self.output = tf.nn.softmax(self.score)
        self.feature = self.fc8
        return self.score

    def get_loss(self, batch_x, batch_y=None):
        with tf.variable_scope('reuse_inference') as scope:
            y_predict = self.inference(batch_x, training=True)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_predict, labels=batch_y))
        return self.loss

    def optimize(self, learning_rate, train_layers):
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[1] in train_layers + ['fc9']]
        finetune_list = [
            v for v in var_list if v.name.split('/')[1] in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
        ]
        new_list = [v for v in var_list if v.name.split('/')[1] in ['fc8', 'fc9']]

        finetune_weights = [v for v in finetune_list if 'weights' in v.name]
        finetune_biases = [v for v in finetune_list if 'biases' in v.name]
        new_weights = [v for v in new_list if 'weights' in v.name]
        new_biases = [v for v in new_list if 'biases' in v.name]

        train_op1 = tf.train.MomentumOptimizer(learning_rate * 0.1, 0.9).minimize(self.loss, var_list=finetune_weights)
        train_op2 = tf.train.MomentumOptimizer(learning_rate * 0.2, 0.9).minimize(self.loss, var_list=finetune_biases)
        train_op3 = tf.train.MomentumOptimizer(learning_rate * 1.0, 0.9).minimize(self.loss, var_list=new_weights)
        train_op4 = tf.train.MomentumOptimizer(learning_rate * 2.0, 0.9).minimize(self.loss, var_list=new_biases)
        train_op = tf.group(train_op1, train_op2, train_op3, train_op4)

        return train_op

    def load_original_weights(self, session, skip_layers=[]):
        weights_dict = np.load('/home/wogong/Models/tensorflow/bvlc_alexnet.npy', encoding='bytes').item()

        for op_name in weights_dict:
            # if op_name in skip_layers:
            #     continue

            if op_name == 'fc8' and self.num_classes != 1000:
                continue

            with tf.variable_scope('reuse_inference/' + op_name, reuse=True):
                for data in weights_dict[op_name]:
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases')
                        #print op_name, var
                        session.run(var.assign(data))
                    else:
                        var = tf.get_variable('weights')
                        #print op_name, var
                        session.run(var.assign(data))


"""
Helper methods
"""


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    input_channels = int(x.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)
        else:
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            conv = tf.concat(axis=3, values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])
        relu = tf.nn.relu(bias, name=scope.name)
        return relu


def fc(x, num_in, num_out, name, relu=True, stddev=0.01):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', initializer=tf.truncated_normal([num_in, num_out], stddev=stddev))
        biases = tf.get_variable('biases', initializer=tf.constant(0.1, shape=[num_out]))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def outer(a, b):
    a = tf.reshape(a, [-1, a.get_shape()[-1], 1])
    b = tf.reshape(b, [-1, 1, b.get_shape()[-1]])
    c = a * b
    return tf.contrib.layers.flatten(c)


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(
        x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)
