import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class model():

	def __init__(self):
		self.graph = tf.Graph().as_default()

		self.batch = 128

		with self.graph():
			iterators = self._get_data(self.batch)

			with tf.variable_scope():
				self._build_graph()


	def _get_data(self, batch):
		train, test = tf.keras.datasets.mnist.load_data()
		train_x, train_y = train
		test_x, test_y = test

		iterator = {}
		def _parse_function(image):
			return image / 255
		def dataset_pipeline(x, y, batch):
			ds = tf.data.Dataset.from_tensor_slices(x)
			ds = ds.map(_parse_function, num_parallel_calls=2)
			dsl = tf.data.Dataset.from_tensor_slices(y)
			ds = tf.data.Dataset.zip((ds, dsl))
			ds = train_ds.shuffle(len(y))
			ds = train_ds.repeat()
			ds = train_ds.batch(batch)
			return ds.make_one_shot_iterator()
		iterators['train'] = dataset_pipeline(train_x, train_y)
		iterators['test'] = dataset_pipeline(test_x, test_y)

		return iterators

	def _build_graph(self, input):

