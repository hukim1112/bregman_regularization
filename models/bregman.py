
import tensorflow as tf
from backbone import inception
from datasets import flower_dataset
from models import bregman_regularizer as bregman
import math
import os
slim = tf.contrib.slim

# params = ['learning_rate', 'batch_size', 'model_dir',
#           'iteration', 'num_classes', 'train_datadir', 'eval_datadir']


class model():

    def __init__(self, params):
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        self.sess = tf.Session(graph=self.graph, config=config)
        self.eval_max = 0
        self.eval_max_global_step = 0
        with self.graph.as_default():
            self.train_model_fn(params)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            variables_to_restore = self.get_init_fn(var_list)

            self.saver = tf.train.Saver(var_list=variables_to_restore)
            self.initializer = tf.global_variables_initializer()

            path_to_latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=os.path.dirname(
                params['model_dir']))
            self.sess.run(self.initializer)

            if params['pretrained_model'] != None:
                self.saver.restore(self.sess, params['pretrained_model'])
                print("restored from pretrained model")
            elif path_to_latest_ckpt != None:
                self.saver.restore(self.sess, path_to_latest_ckpt)
                print("restored from latest checkpoint")
            else:
                print("scratch from random distribution")

    def train(self, params):
        try:
            for i in range(1, params['iteration'] + 1):
                if i != 1:
                    _, loss, __, global_step = self.sess.run(
                        [self.solver, self.loss, self.update_prototypes, self.global_step])
                else:
                    _, global_step = self.sess.run(
                        [self.update_prototypes, self.global_step])
                if i % 10 == 0:
                    print("iteration {} : loss={}".format(global_step, loss))
                if i % 500 == 0:
                    score = self.eval(params)
                    print("evaluation accuracy {}".format(score))
                    if self.eval_max < score:
                        self.eval_max = score
                        self.eval_max_global_step = global_step
                        self.saver.save(
                            self.sess, params['model_dir'], global_step=self.global_step)
                    tf.summary.scalar('eval_accuracy', score)

                if i == params['iteration']:
                    print(self.eval_max, self.eval_max_global_step)
        except KeyboardInterrupt:
            print('Got Keyboard Interrupt, saving model and close')
            self.saver.save(
                self.sess, params['model_dir'], global_step=self.global_step)

    def eval(self, params):
        # Initialize counter at each epoch
        self.sess.run(self.eval_dataset_iterator.initializer)
        self.sess.run(self.running_vars_initializer)

        for i in range(self.batch_per_epoch):
            self.sess.run(self.update_op)
        score = self.sess.run(self.accuracy)
        return score

    def load_batch(self, dataset_dir, batch_size, mode='training'):
        if mode == 'training':
            num_images = len(filepaths)
            filepaths, class_name_to_ids = flower_dataset.load_data(
                dataset_dir)
            dataset = flower_dataset.input_fn(
                filepaths, class_name_to_ids, batch_size, num_images, mode)
            iterator = dataset.make_one_shot_iterator()
            return iterator, num_images

        elif mode == 'categoy_balanced_input':
            filepaths, class_name_to_ids = flower_dataset.load_categorical_data(
                dataset_dir)
            categorical_batch_size = math.floor(
                batch_size / len(class_name_to_ids.keys()))
            category_balanced_data = flower_dataset.category_balancing_input_fn(
                filepaths, class_name_to_ids, categorical_batch_size)
            return category_balanced_data, categorical_batch_size
        # elif mode == 'predict':
        #     filepaths, class_name_to_ids = flower_dataset.load_data(
        #         dataset_dir)
        #     num_images = len(filepaths)
        #     dataset = flower_dataset.predict_input_fn(
        #         filepaths, class_name_to_ids, batch_size)

        else:  # 'eval'
            filepaths, class_name_to_ids = flower_dataset.load_data(
                dataset_dir)
            num_images = len(filepaths)
            dataset = flower_dataset.input_fn(
                filepaths, class_name_to_ids, batch_size, num_images, mode)
            iterator = dataset.make_initializable_iterator()
            return iterator, num_images

    def inference(self, images, params, reuse=None):
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, endpoints, scope = inception.inception_v1(
                images, num_classes=params['num_classes'], is_training=True, reuse=reuse)
        return logits, endpoints, scope

    def train_model_fn(self, params):

        # get batch of training dataset
        category_balanced_data, categorical_batch_size = self.load_batch(
            params['train_datadir'], params['batch_size'], mode='categoy_balanced_input')
        print("Each batch we get {} images for each category".format(
            categorical_batch_size))
        images = tf.concat([x[0]
                            for x in category_balanced_data.values()], axis=0)
        labels = tf.concat([x[1]
                            for x in category_balanced_data.values()], axis=0)

        # get a batch of evaluation dataset
        self.eval_dataset_iterator, num_eval_images = self.load_batch(
            params['eval_datadir'], params['batch_size'], mode='eval')
        eval_images, eval_labels = self.eval_dataset_iterator.get_next()
        self.batch_per_epoch = math.ceil(
            num_eval_images / params['batch_size'])

        # bring our model for training
        logits, endpoints, scope = self.inference(images, params)
        embeddings = endpoints['AvgPool_0a_7x7']
        embeddings = tf.squeeze(embeddings, [1, 2], name='SpatialSqueeze')

        model_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        self.global_step = tf.train.get_or_create_global_step()

        # Cross-entropy optimization
        #one_hot_labels = tf.one_hot(labels, depth=params['num_classes'])
        #cross_entropy_loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)

        # Bregman regularization optimization
        prototypes = tf.get_variable('prototypes', shape=(
            params['num_classes'], embeddings.shape[1]), dtype=tf.float32, trainable=False)
        #bregman_loss = bregman.get_bregman_loss_from_embeddings(embeddings, labels, prototypes, params['num_classes'])
        classifier_scores = bregman.prototypical_classifier(
            embeddings, labels, prototypes, params['num_classes'])
        one_hot_labels = tf.one_hot(labels, depth=params['num_classes'])
        cross_entropy_loss = tf.losses.softmax_cross_entropy(
            one_hot_labels, classifier_scores)

        new_prototypes = bregman.get_prototypes_from_embeddings(
            embeddings, labels, params['num_classes'])
        self.update_prototypes = tf.assign(prototypes, new_prototypes)

        self.loss = cross_entropy_loss
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate'])
        self.solver = optimizer.minimize(
            self.loss, global_step=self.global_step, var_list=model_var)
        tf.summary.scalar('bregman_loss', self.loss)

        # build a graph for evaluation. Reuse our model for evaluating
        _, eval_endpoints, __ = self.inference(
            eval_images, params, reuse=tf.AUTO_REUSE)
        eval_embeddings = eval_endpoints['AvgPool_0a_7x7']
        eval_embeddings = tf.squeeze(
            eval_embeddings, [1, 2], name='SpatialSqueeze')
        classifier_scores = bregman.prototypical_classifier(
            eval_embeddings, labels, prototypes, params['num_classes'])

        predicted_indices = tf.argmax(input=classifier_scores, axis=1)
        #probabilities = tf.nn.softmax(logits, name='soft_tensor')
        self.accuracy, self.update_op = tf.metrics.accuracy(labels=eval_labels,
                                                            predictions=predicted_indices,
                                                            name='acc_op')
        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="acc_op")
        # Define initializer to initialize/reset running variables
        self.running_vars_initializer = tf.variables_initializer(
            var_list=running_vars)

        return

    def get_init_fn(self, var_list):
        """Returns a function run by the chief worker to warm-start the training."""
        checkpoint_exclude_scopes = [
            "InceptionV1/Logits", "InceptionV1/AuxLogits"]

        exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

        variables_to_restore = []
        for var in var_list:
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        return variables_to_restore
