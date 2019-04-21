import data_util
import tensorflow as tf

dataset = util.batch_random(128, 'train')

iterator = dataset.make_one_shot_iterator()
element = iterator.get_next()

sess = tf.Session()

x = sess.run(element)
print(x.shape)
