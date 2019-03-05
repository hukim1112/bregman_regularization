
import tensorflow as tf
from backbone import inception
from datasets import flower_dataset
import math
slim = tf.contrib.slim

# params = ['learning_rate', 'batch_size', 'model_dir',
#           'iteration', 'num_classes', 'train_datadir', 'eval_datadir']


class model():

    def __init__(self, params):
        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            self.train_model_fn(params)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            variables_to_restore = self.get_init_fn(var_list)

            print(variables_to_restore)

            self.saver = tf.train.Saver(var_list=variables_to_restore)
            self.initializer = tf.global_variables_initializer()

            path_to_latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir=params['model_dir'])
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
            for i in range(params['iteration']):
                _, loss, global_step = self.sess.run([self.cross_entropy_solver, self.loss, self.global_step])
                if i % 10:
                    print("iteration {} : loss={}".format(global_step, loss))
                if i % 1000:
                    score = self.eval(params)
                    print("evaluation accuracy {}".format(score))
                    tf.summary.scalar('eval_accuracy', score)
                    self.saver.save(self.sess, params['model_dir'], global_step=self.global_step)
        except KeyboardInterrupt:
            print('Got Keyboard Interrupt, saving model and close')
            self.saver.save(self.sess, params['model_dir'], global_step=self.global_step)


    def eval(self, params):
        # Initialize counter at each epoch
        self.sess.run(self.running_vars_initializer)

        for i in range(self.batch_per_epoch):
            self.sess.run(self.update_op)

        score = self.sess.run(self.accuracy)
        return score

    def load_batch(self, dataset_dir, batch_size, mode='training'):
        filepaths, class_name_to_ids = flower_dataset.load_data(dataset_dir)
        num_images = len(filepaths)
        if mode == 'predict':
            dataset = flower_dataset.predict_input_fn(
                filepaths, class_name_to_ids, batch_size)
        else:
            dataset = flower_dataset.input_fn(
                filepaths, class_name_to_ids, batch_size, num_images, mode)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()
        return images, labels, num_images

    def inference(self, images, params, reuse=None):
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, endpoints, scope= inception.inception_v1(
                images, num_classes=params['num_classes'], is_training=True, reuse=reuse)
        return logits, scope

    def train_model_fn(self, params):
        
        images, labels, _ = self.load_batch(
            params['train_datadir'], params['batch_size'], mode='training')

        eval_images, eval_labels, num_eval_images = self.load_batch(
                params['eval_datadir'], params['batch_size'], mode='eval')
        self.batch_per_epoch = math.ceil(num_eval_images/params['batch_size'])

        # bring our model for training
        logits, scope = self.inference(images, params)
            
        # Reuse our model for evaluating
        eval_logits, scope = self.inference(eval_images, params, reuse=tf.AUTO_REUSE)
        predicted_indices = tf.argmax(input=eval_logits, axis=1)
        #probabilities = tf.nn.softmax(logits, name='soft_tensor')
        self.accuracy, self.update_op = tf.metrics.accuracy(labels=eval_labels,
                                                      predictions=predicted_indices,
                                                      name='acc_op')
        # Isolate the variables stored behind the scenes by the metric operation
        running_vars = tf.get_collection(
                tf.GraphKeys.LOCAL_VARIABLES, scope="acc_op")
        # Define initializer to initialize/reset running variables
        self.running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        model_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        self.global_step = tf.train.get_or_create_global_step()
        one_hot_labels = tf.one_hot(labels, depth=params['num_classes'])
        self.loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=params['learning_rate'])
        self.cross_entropy_solver = optimizer.minimize(
            self.loss, global_step=self.global_step, var_list=model_var)
        tf.summary.scalar('cross_entropy', self.loss)
        return

    def get_init_fn(self, var_list):
        """Returns a function run by the chief worker to warm-start the training."""
        checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits"]
        
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