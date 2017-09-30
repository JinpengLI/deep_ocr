# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:18:36 2017

@author: jinpeng.li
"""
from __future__ import division

import math
import copy
import six
import random
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
import numpy as np
import tensorflow as tf
import cv2
import keras

from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)
    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[0], input_shape[1], input_shape[2])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


class KerasModel(object):
    def __init__(self, norm_shape=(28, 28), max_n_label=10, batch_size=32,
                 verbose=1, nb_epoch=10):
        self.verbose = verbose
        self.norm_shape = norm_shape
        self.batch_size = batch_size
        self.model = None
        self.max_n_label = max_n_label
        self.nb_epoch = nb_epoch

    def fit(self, X, y):
        early_stopping = keras.callbacks.EarlyStopping(monitor="loss")
        callbacks = [
            early_stopping
        ]

        self.model.fit(X, y,
                       batch_size=self.batch_size,
                       epochs=self.nb_epoch,
                       verbose=self.verbose,
                       callbacks=callbacks)

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, verbose=0)
        return score

    def config_str(self, ):
        tmp_config = "norm_shape=" + str(self.norm_shape) + \
            ", batch_size=" + str(self.batch_size) + \
            ", nb_epoch=" + str(self.nb_epoch)
        return "("+ tmp_config + ')'

    def __repr__(self, ):
        return "KerasModel"

class KerasLenetModel(KerasModel):
    def __init__(self, **kwargs):
        super(KerasLenetModel, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        self.model = Sequential()
        self.model.add(Convolution2D(32, (3, 3), activation='relu',
                                input_shape=(norm_shape[0], norm_shape[1], 1)))
        self.model.add(Convolution2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.max_n_label, activation='softmax'))

        # 8. Compile model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def __repr__(self, ):
        return "KerasLenetMode" + self.config_str()


class KerasLenetExt1Model(KerasModel):
    def __init__(self, **kwargs):
        super(KerasLenetExt1Model, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        self.model = Sequential()
        self.model.add(Convolution2D(64, (3, 3), activation='relu',
                                input_shape=(norm_shape[0], norm_shape[1], 1)))

        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.max_n_label, activation='softmax'))

        # 8. Compile model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def __repr__(self, ):
        return "KerasLenetExt1Model" + self.config_str()

class KerasLenetExt2Model(KerasModel):
    def __init__(self, **kwargs):
        self.final_fc_nm = kwargs.get("final_fc_nm", 2048)
        if "final_fc_nm" in kwargs:
            kwargs.pop("final_fc_nm")
        super(KerasLenetExt2Model, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        self.model = Sequential()
        self.model.add(Convolution2D(64, (3, 3), activation='relu',
                                input_shape=(norm_shape[0], norm_shape[1], 1)))

        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Convolution2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(self.final_fc_nm, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.max_n_label, activation='softmax'))

        # 8. Compile model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    def __repr__(self, ):
        return "KerasLenetExt2Model" + self.config_str()


class KerasCifar10CNN(KerasModel):

    def __init__(self, **kwargs):
        super(KerasCifar10CNN, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(norm_shape[0], norm_shape[1], 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', ))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.max_n_label))
        model.add(Activation('softmax'))
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
              metrics=['accuracy'])
        self.model = model

    def __repr__(self, ):
        return "KerasCifar10CNN" + self.config_str()

class KerasCifar10CNN2(KerasModel):

    def __init__(self, **kwargs):
        super(KerasCifar10CNN2, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        model = Sequential()
        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=(norm_shape[0], norm_shape[1], 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same', ))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), padding='same', ))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3), ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(2048))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.max_n_label))
        model.add(Activation('softmax'))
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
              metrics=['accuracy'])
        self.model = model

    def __repr__(self, ):
        return "KerasCifar10CNN2" + self.config_str()


class KerasVGG16(KerasModel):
    def __init__(self, **kwargs):
        super(KerasVGG16, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        model = Sequential()
        model.add(ZeroPadding2D((1,1), input_shape=(norm_shape[0], norm_shape[1], 1),
                                ))
        model.add(Convolution2D(64, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(64, (3, 3), activation='relu', ))
        model.add(MaxPooling2D((2,2), strides=(2,2), ))

        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(128, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(128, (3, 3), activation='relu', ))
        model.add(MaxPooling2D((2,2), strides=(2,2), ))

        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(256, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(256, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(256, (3, 3), activation='relu', ))
        model.add(MaxPooling2D((2,2), strides=(2,2), ))

        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(512, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(512, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(512, (3, 3), activation='relu', ))
        model.add(MaxPooling2D((2,2), strides=(2,2), ))

        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(512, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(512, (3, 3), activation='relu', ))
        model.add(ZeroPadding2D((1,1), ))
        model.add(Convolution2D(512, (3, 3), activation='relu', ))
        model.add(MaxPooling2D((2,2), strides=(2,2), ))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.max_n_label, activation='softmax'))
        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
              metrics=['accuracy'])
        self.model = model

    def __repr__(self, ):
        return "KerasVGG16" + self.config_str()


class KerasResNet(KerasModel):
    def __init__(self, **kwargs):
        self.nw_type = kwargs.get("nw_type", 18)
        if "nw_type" in kwargs:
            kwargs.pop("nw_type")
        super(KerasResNet, self).__init__(**kwargs)
        norm_shape = self.norm_shape
        nb_classes = self.max_n_label
        if self.nw_type == 18:
            model = ResnetBuilder.build_resnet_18(
                (norm_shape[0], norm_shape[1], 1), nb_classes)
        elif self.nw_type == 34:
            model = ResnetBuilder.build_resnet_34(
                (norm_shape[0], norm_shape[1], 1), nb_classes)
        elif self.nw_type == 50:
            model = ResnetBuilder.build_resnet_50(
                (norm_shape[0], norm_shape[1], 1), nb_classes)
        elif self.nw_type == 101:
            model = ResnetBuilder.build_resnet_101(
                (norm_shape[0], norm_shape[1], 1), nb_classes)
        elif self.nw_type == 152:
            model = ResnetBuilder.build_resnet_152(
                (norm_shape[0], norm_shape[1], 1), nb_classes)
        else:
            raise ValueError("unknown network.")
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model
    def __repr__(self, ):
        return "KerasResNet" + self.config_str() \
            + "(nw_type=" + str(self.nw_type) + ")"

def load_image(image_path, norm_shape=(28, 28)):
    image = cv2.imread(image_path, 0)
    resized_img = cv2.resize(image, norm_shape,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img

def load_dataset(image_label_path, norm_shape=(28, 28)):
    images = []
    labels = []
    for line in open(image_label_path, "r"):
        words = line.split()
        if len(words) < 2:
            continue
        image_path = words[0]
        label = words[1]
        images.append(load_image(image_path, norm_shape))
        labels.append(int(label))
    images = np.asarray(images, dtype=np.float32)
    images = images / 255.0
    max_n_label = max(labels) + 1
    labels = np.asarray(labels, dtype=np.uint8)
    return images, labels, max_n_label

def generate_selected_items(max_epoch_batch, n_items):
    if max_epoch_batch >= n_items:
        yield range(n_items)
        return
    for i in range(max_epoch_batch, n_items, max_epoch_batch):
        yield range(i - max_epoch_batch, i)
    yield range(n_items)[-max_epoch_batch:]

if __name__ == "__main__":
    path_train = "train.txt"
    path_test = "test.txt"
    verbose = 1
    max_epoch_batch = 1400000 ## set it smaller when there is not enough memeory
    nrepeat = 3
    configs = [
        #{"model": KerasLenetModel, "norm_shape": (22, 22), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasLenetModel, "norm_shape": (26, 26), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasLenetModel, "norm_shape": (28, 28), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasLenetModel, "norm_shape": (30, 30), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasLenetModel, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasCifar10CNN, "norm_shape": (30, 30), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasCifar10CNN, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasVGG16, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasResNet, "norm_shape": (32, 32), "batch_size": 128, "nb_epoch": 150},
        #{"model": KerasResNet, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150, "nw_type": 18, },
        #{"model": KerasResNet, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150, "nw_type": 34, },
        #{"model": KerasResNet, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150, "nw_type": 50, },
        #{"model": KerasResNet, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150, "nw_type": 101, },
        #{"model": KerasResNet, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150, "nw_type": 152, },
        #{"model": KerasLenetExt1Model, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150,  }, ## 0.925 on test
        #{"model": KerasLenetExt2Model, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150,  "final_fc_nm": 2048}, ## 0.949 and 0.941 on test
        #{"model": KerasLenetExt2Model, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150,  "final_fc_nm": 4096}, ## 0.944, 0.92651, 0.9454
        #{"model": KerasLenetExt2Model, "norm_shape": (32, 32), "batch_size": 128, "nb_epoch": 150,  "final_fc_nm": 2048},  ## 0.94287295149762418
        #{"model": KerasLenetExt3Model, "norm_shape": (64, 64), "batch_size": 128, "nb_epoch": 150,  },
        #{"model": KerasCifar10CNN2, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150, }, ## 0.87882, 0.882637, 0.878889
        {"model": KerasLenetExt2Model, "norm_shape": (36, 36), "batch_size": 128, "nb_epoch": 150,  "final_fc_nm": 10240}
    ]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45
    set_session(tf.Session(config=config))

    results = {}
    for tune_config in configs:
        for i_repect in range(nrepeat):
            repeat_tune_config = copy.deepcopy(tune_config)
            norm_shape = repeat_tune_config["norm_shape"]
            TuneModel = repeat_tune_config["model"]
            repeat_tune_config.pop("model")
            X_train, y_train, max_n_label = load_dataset(path_train, norm_shape)
            X_test, y_test, _ = load_dataset(path_test, norm_shape)
            # 5. Preprocess input data
            X_train = X_train.reshape(X_train.shape[0], norm_shape[0], norm_shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], norm_shape[0], norm_shape[1], 1)
            repeat_tune_config["max_n_label"] = max_n_label
            print("trainning data #item ", X_train.shape[0])
            print("test data #item ", X_test.shape[0])
            cur_model = TuneModel(**repeat_tune_config)

            n_items = X_train.shape[0]
            n_times = max(int(math.ceil(n_items/max_epoch_batch)), 1)

            print("split dataset into n_times=", n_times)
            i_times = 0
            for seleted_items in generate_selected_items(max_epoch_batch, n_items):
                print("="*40)
                print("current time %d/%d" % (i_times, n_times))
                selected_X_train = X_train[seleted_items, :]
                selected_y_train = y_train[seleted_items]
                # 6. Preprocess class labels
                selected_Y_train = np_utils.to_categorical(selected_y_train, max_n_label)
                # 9. Fit model on training data
                cur_model.fit(selected_X_train, selected_Y_train)
                del selected_X_train, selected_Y_train, selected_y_train
                i_times += 1

            Y_test = np_utils.to_categorical(y_test, max_n_label)
            # 10. Evaluate model on test data
            score = cur_model.evaluate(X_test, Y_test)
            print(repr(cur_model))
            print("score on test:", score)
            result_key = repr(cur_model)
            if result_key not in results:
                results[result_key] = []
            results[result_key].append(score[1])
            del X_train, y_train, X_test, y_test, Y_test
            del cur_model
    final_resuts = []
    for result_key in results:
        avg_res = np.average(results[result_key])
        final_resuts.append((result_key, avg_res))
    results = final_resuts
    results = sorted(results, key=lambda x:x[1])
    for result_pair in results:
        print("config %s result: %f" % (result_pair[0], result_pair[1]))




