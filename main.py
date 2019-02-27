import tensorflow as tf
from datasets import datasets
from models import models

tf.enable_eager_execution()

class FLAGS():
	pass

#parameters
FLAGS.batch_size=64
FLAGS.max_steps=20000
FLAGS.eval_step=5
FLAGS.save_checkpoints_steps=1000
FLAGS.learning_rate=0.001
FLAGS.num_classes=5

#PATHs
FLAGS.model_dir='/home/dan/prj/checkpoints/flowers/vanilla'
FLAGS.train_datadir='/home/dan/prj/datasets/flowers/train'
FLAGS.eval_datadir='/home/dan/prj/datasets/flowers/eval'
FLAGS.warm_start_from='/home/dan/prj/checkpoints/inception_v1'

#Image info
FLAGS.IMAGE_HEIGHT = 224
FLAGS.IMAGE_WIDTH = 224
FLAGS.IMAGE_DEPTH = 3

def main():
	params = {'feature_columns' : None, 
			  'num_classes' : FLAGS.num_classes, 
			  'learning_rate' : FLAGS.learning_rate,
			  'batch_size' : FLAGS.batch_size,
			  'IMAGE_HEIGHT' : FLAGS.IMAGE_HEIGHT,
			  'IMAGE_WIDTH' : FLAGS.IMAGE_WIDTH,
			  'IMAGE_DEPTH' : FLAGS.IMAGE_DEPTH}

	params['feature_columns'] = datasets.get_feature_columns(FLAGS.IMAGE_HEIGHT, FLAGS.IMAGE_WIDTH, FLAGS.IMAGE_DEPTH)

	run_config=tf.estimator.RunConfig(
		save_checkpoints_steps=FLAGS.save_checkpoints_steps,
		model_dir=FLAGS.model_dir)

	train_filepaths, class_names_to_ids = datasets.load_data(FLAGS.train_datadir)
	eval_filepaths, class_names_to_ids = datasets.load_data(FLAGS.eval_datadir)
	datasets.input_fn(train_filepaths, class_names_to_ids, batch_size=params['batch_size'])
	classifier = tf.estimator.Estimator(model_fn=models.model_fn,
										config=run_config,
										params=params,
										warm_start_from=FLAGS.warm_start_from
										)

	train_spec = tf.estimator.TrainSpec(input_fn=lambda:datasets.input_fn(train_filepaths, class_names_to_ids, params['batch_size']), max_steps=FLAGS.max_steps)
	eval_spec = tf.estimator.EvalSpec(input_fn=lambda:datasets.input_fn(eval_filepaths, class_names_to_ids, params['batch_size'], step=FLAGS.eval_step))

	tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	main()


