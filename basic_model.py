import tensorflow as tf
import os


class basic_model():
    def __init__(self, config):
        self.config = config
        self.graph = tf.graph().as_default()
        self.datasets = ['train', 'eval']

        with self.graph:

            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                self._build_graph()

    def _build_dataset(self, mode):
        raise NotImplementedError

    def _model(self, inputs, mode, **config):
        raise NotImplementedError

    def _loss(self, outputs, **config):
        raise NotImplementedError

    def _build_graph(self):

        # Training and evaluation network, each dataset is given.
        for n in self.datasets:
            train_images, train_labels = _build_dataset(n):
            


    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
