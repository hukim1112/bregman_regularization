
import tensorflow as tf
from backbone import inception
from datasets import flower_dataset
slim = tf.contrib.slim

params = ['learning_rate', '']


class model():

    def __init__(self):
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            self.model_fn(params)
            self.saver = tf.train.Saver()
            self.initializer = tf.global_variables_initializer()

            path_to_latest_ckpt = tf.train.latest_checkpoint(
                checkpoint_dir=params['model_dir'])
            if path_to_latest_ckpt == None:
                print('scratch from random distribution')
                self.sess.run(self.initializer)
            else:
                self.saver.restore(self.sess, path_to_latest_ckpt)
                print('restored')

    def train(self, params):

    def eval(self, params):

    def load_batch(self, dataset_dir, batch_size):
        filepaths, class_name_to_ids = flower_dataset.load_data(dataset_dir)
        dataset = flower_dataset.input_fn(
            filepaths, class_name_to_ids, batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        return images, labels

    def inference(self, images, params):
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, endpoints = inception.inception_v1(
                images, num_classes=params['num_classes'], is_training=True)
        return logits

    def model_fn(self, params):
        # bring our model for training
        images, labels = self.load_batch(
            params['train_datadir'], params['batch_size'])
        with tf.variable_scope("model") as self.model_scope:
            logits = self.inference(images, params)

        global_step = tf.train.get_or_create_global_step()
        one_hot_labels = exitf.one_hot(labels, depth=params['num_classes'])
        loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate'])
        self.cross_entropy_solver = optimizer.minimize(
            loss, global_step=global_step, var_list=self.model_scope)
        tf.summary.scalar('cross_entropy', loss)

        # Reuse our model for evaluating
        eval_images, eval_labels = self.load_batch(
            params['eval_datadir'], params['batch_size'])
        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            eval_logits = self.inference(eval_images, params)
        predicted_indices = tf.argmax(input=eval_logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='soft_tensor')
