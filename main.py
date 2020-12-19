"""
  Created on Monday December 7 2020 at 2:55 p.m.
  Author Keisuke Noji @CVSLab.

  Convolutional Neural Network using MNIST dataset
"""


import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt


class InitialParameter:
    def __init__(self):
        # Divide Train and Validation
        self.spirit_size = 0.2

        # Number Of Data
        self.train_num = ((self.spirit_size.__rsub__(1)).__mul__(60000)).__int__()
        self.test_num = (self.spirit_size.__mul__(60000)).__int__()
        self.predict_num = 10000

        # filter Initial Num
        # Ex) Layer1: 16, Layer2: 32, ...
        self.filter_initial = 16

        # Image Size
        self.sample = (None, 28, 28, 1)

        # Batch Size
        self.batch_size = 50

        # Epoch
        self.epoch = 5

        # Learning Rate
        self.lr = 1e-03

        # Optimizer
        self.optimizer = tf.optimizers.Adam(lr=self.lr)

        # Loss Function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


class DataLoader(InitialParameter):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.eight_bit_max = 255.0

    def __call__(self, batch_size=None, training=False, spirit_mode=False):
        data, label = self.load_mnist(training=training)

        if spirit_mode is True:
            data1_ds, data2_ds = self.dataset(x=data, label=label,
                                              batch_size=batch_size,
                                              training=training,
                                              spirit_mode=spirit_mode)
            return data1_ds, data2_ds

        else:
            data_ds = self.dataset(x=data, label=label,
                                   batch_size=batch_size,
                                   training=training)
            return data_ds

    def load_mnist(self, training=True):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = tf.cast(x_train, tf.float32).__truediv__(self.eight_bit_max)
        x_train = x_train[..., tf.newaxis]

        x_test = tf.cast(x_test, tf.float32).__truediv__(self.eight_bit_max)
        x_test = x_test[..., tf.newaxis]

        if training is True:
            return x_train, y_train
        else:
            return x_test, y_test

    def dataset(self, x=None, label=None, batch_size=None,
                training=False, spirit_mode=False):
        if batch_size is None:
            batch_size = self.batch_size

        x_ds = tf.data.Dataset.from_tensor_slices((x, label))
        if training is True:
            x_ds = x_ds.shuffle(x.__len__())
        if spirit_mode is True:
            x_ds, y_ds = self.two_spirit(num=x.__len__(),
                                         x_ds=x_ds,
                                         spirit_size=self.spirit_size)
            x_ds = x_ds.batch(batch_size)
            y_ds = y_ds.batch(batch_size)
            return x_ds, y_ds
        else:
            x_ds = x_ds.batch(batch_size)
            return x_ds

    def two_spirit(self, num, x_ds, spirit_size):
        train_size = ((spirit_size.__rsub__(1)).__mul__(num)).__int__()
        print('{:20}'.format('train number : '), train_size)
        test_size = (spirit_size.__mul__(num)).__int__()
        print('{:20}'.format('test  number : '), test_size)
        x_ds = x_ds.take(train_size)
        y_ds = x_ds.skip(train_size)
        y_ds = x_ds.take(test_size)
        return x_ds, y_ds


class ConvolutionSet(tf.keras.Model):
    def __init__(self,
                 filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='valid',
                 kernel_initializer=tf.keras.initializers.he_normal,
                 activation='elu',
                 pool_size=(2, 2),
                 **kwargs):
        super(ConvolutionSet, self).__init__(**kwargs)
        self.convolution = tf.keras.layers.Conv2D(filters=filters,
                                                  kernel_size=kernel_size,
                                                  strides=strides,
                                                  padding=padding,
                                                  kernel_initializer=kernel_initializer)
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation=activation)
        self.max_pooling = tf.keras.layers.MaxPool2D(pool_size=pool_size)

    def call(self, inputs, training=None, mask=None):
        x = self.convolution(inputs)
        x = self.batch_normalization(x, training=training)
        x = self.activation(x)
        x = self.max_pooling(x)
        return x


class FullConnectedSet(tf.keras.Model):
    def __init__(self,
                 units=32,
                 activation='linear',
                 **kwargs):
        super(FullConnectedSet, self).__init__(**kwargs)

        self.fc = tf.keras.layers.Dense(units=units)
        self.activation = tf.keras.layers.Activation(activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = self.fc(inputs)
        x = self.activation(x)
        return x


class ClassificationNetwork(tf.keras.Model):
    def __init__(self,
                 filter_initial):
        super(ClassificationNetwork, self).__init__()
        self.bit = 2
        self._conv1 = ConvolutionSet(filters=filter_initial.__mul__(self.bit.__pow__(0)),
                                     name='xConv_1')
        self._conv2 = ConvolutionSet(filters=filter_initial.__mul__(self.bit.__pow__(1)),
                                     name='xConv_2',
                                     padding='same')
        self._conv3 = ConvolutionSet(filters=filter_initial.__mul__(self.bit.__pow__(2)),
                                     name='xConv_3')
        self._flatten = tf.keras.layers.Flatten()
        self._fc_1 = FullConnectedSet(units=32, activation='relu', name='xFC_1')
        self._fc_2 = FullConnectedSet(units=10, activation='softmax', name='xOutput')

    def call(self, inputs, training=None, mask=None):
        x = self._conv1(inputs, training=training)
        x = self._conv2(x, training=training)
        x = self._conv3(x, training=training)
        x = self._flatten(x)
        x = self._fc_1(x)
        return self._fc_2(x)

    def build_graph(self, input_shape):
        input_shape_no_batch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.layers.Input(shape=input_shape_no_batch)
        self.call(inputs=inputs, training=False, mask=None)


class LearningAndPredict(InitialParameter):
    def __init__(self):
        super(LearningAndPredict, self).__init__()
        self.model = ClassificationNetwork(filter_initial=self.filter_initial)
        self.train_accuracy = \
            tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_epoch_loss = tf.metrics.Mean(name='train_epoch_loss')
        self.test_epoch_loss = tf.metrics.Mean(name='test_epoch_loss')
        self.test_accuracy = \
            tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.predict_accuracy = \
            tf.keras.metrics.SparseCategoricalAccuracy(name='predict_accuracy')

        self.train_acc_list = list()
        self.test_acc_list = list()
        self.train_loss_list = list()
        self.test_loss_list = list()

    @tf.function
    def train_on_batch(self, image, label):
        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss = self.loss_object(label, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, self.train_accuracy(label, predictions)

    @tf.function
    def test_on_batch(self, image, label, validation=False):
        predictions = self.model(image, training=False)
        loss = self.loss_object(label, predictions)
        if validation is True:
            accuracy = self.test_accuracy(label, predictions)
        else:
            accuracy = self.predict_accuracy(label, predictions)
        return loss, accuracy

    def train_log(self, i=None, training=False):
        if training is True:
            template = 'Epoch {:>4}, Loss: {:.4f}, Accuracy: {:.4f}, ' \
                       'Test Loss: {:.4f}, Test Accuracy: {:.4f}'
            print(template.format(i.__add__(1),
                                  self.train_epoch_loss.result(),
                                  self.train_accuracy.result(),
                                  self.test_epoch_loss.result(),
                                  self.test_accuracy.result()))
        else:
            template = 'Predict Accuracy: {:.4f}'
            print(template.format(self.predict_accuracy.result()))

    def plot_figure(self, train=None, test=None, title=None):
        epochs = np.array(range(self.epoch))
        plt.plot(epochs, train, color='blue', label='train')
        plt.plot(epochs, test, color='orange', label='validation')

        plt.title(title),
        plt.xlabel('epochs')
        plt.ylabel(title)
        plt.legend()

    def learn_curve(self):
        sns.set()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        self.plot_figure(train=self.train_acc_list,
                         test=self.test_acc_list,
                         title='Accuracy')

        plt.subplot(1, 2, 2)
        self.plot_figure(train=self.train_loss_list,
                         test=self.test_loss_list,
                         title='Loss')
        plt.show()

    def learn(self, i=None, data_ds=None, training=False):
        if training is True:
            with tqdm(total=self.train_num, unit_scale=True,
                      ascii=False, miniters=1) as bar:
                bar.set_description(
                    'train epoch {}/{}'.format(i, self.epoch))

                for images, label in data_ds:
                    train_loss, acc = \
                        self.train_on_batch(image=images, label=label)

                    self.train_epoch_loss.update_state(train_loss)

                    bar.set_postfix(OrderedDict(
                        train_accuracy=acc.numpy(),
                        train_loss=self.train_epoch_loss.result().numpy()
                    ))
                    bar.update(self.batch_size)

                self.train_acc_list.append(acc.numpy())
                self.train_loss_list.append(train_loss.numpy())
        else:
            with tqdm(total=self.test_num, unit_scale=True,
                      ascii=False, miniters=1) as bar:
                bar.set_description(
                    'test  epoch {}/{}'.format(i, self.epoch))

                for images, label in data_ds:
                    test_loss, acc = self.test_on_batch(image=images, label=label,
                                                        validation=True)
                    self.test_epoch_loss.update_state(test_loss)

                    bar.set_postfix(OrderedDict(
                        test_acc=acc.numpy(),
                        test_epoch_loss=self.test_epoch_loss.result().numpy()
                    ))

                    bar.update(self.batch_size)

                self.test_acc_list.append(acc.numpy())
                self.test_loss_list.append(test_loss.numpy())

    def train_step(self, train_ds, test_ds):
        for i in range(self.epoch):
            self.learn(i=i, data_ds=train_ds, training=True)
            self.learn(i=i, data_ds=test_ds, training=False)
            self.train_log(i=i, training=True)
        self.learn_curve()
        self.train_accuracy.reset_states()
        self.train_epoch_loss.reset_states()
        self.test_accuracy.reset_states()
        self.test_epoch_loss.reset_states()

    def prediction(self, data_ds, data_num):
        count = 1
        with tqdm(total=data_num) as bar:
            for image, label in data_ds:
                bar.set_description(
                    'predict image {}/{}'.format(count, data_num))
                _, acc = self.test_on_batch(image=image, label=label,
                                            validation=False)
                bar.update(1)
                count = count.__add__(1)
        self.train_log(i=None, training=False)
        self.predict_accuracy.reset_states()


class NeuralNetwork(LearningAndPredict):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def __call__(self, *args, **kwargs):
        """ Data Load"""
        loader = DataLoader()
        train_ds, test_ds = loader(training=True, spirit_mode=True)
        predict_ds = loader(batch_size=1, training=False, spirit_mode=False)

        """ Network Load """
        self.model.build_graph(self.sample)
        self.model.summary()

        """ Train """
        self.train_step(train_ds=train_ds, test_ds=test_ds)

        """ Test """
        self.prediction(data_ds=predict_ds, data_num=self.predict_num)
        print('Done')


def main():
    cnn = NeuralNetwork()
    cnn()


if __name__ == '__main__':
    main()
