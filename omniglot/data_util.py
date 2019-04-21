import os
import tensorflow as tf
data_generator_path = os.environ['DATA_GENERATOR']

def normalize(images):
    """
    Normalize images to [-1,1]
    """

    images = tf.cast(images, tf.float32)
    images /= 255. # 0 ~ 1
    images -= 0.5  # -0.5 ~ 0.5
    images *= 2 # -1 ~ 1
    return images

def _parse_function(filename, rot90, is_training=True):
    image_string = tf.cast(filename, tf.string)
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image_decoded = tf.image.resize_images(image_decoded, (28, 28))
    image = normalize(image_decoded)
    image = tf.image.rot90(image, k=rot90, name=None)
    return image

def batch_random(batch_size, split):
    assert split in ['train', 'test', 'trainval']
    split_path = os.path.join(data_generator_path, 'labels/omniglot', split+'.txt')
    with open(split_path, 'r') as split:
        classes = [line.rstrip() for line in split.readlines()]

    omniglot = []
    rot90 = []
    for line in classes:
        class_name = os.path.dirname(line)
        rotation = os.path.basename(line)
        dir_path = os.path.join(data_generator_path, 'datasets', 'omniglot', class_name)
        if os.path.isdir(dir_path):
            file_list = []
            for _filename in os.listdir(dir_path):
                omniglot.append( os.path.join(dir_path, _filename) )
                rot90.append(int(rotation[3:6])//90)


    dataset = tf.data.Dataset.from_tensor_slices((omniglot, rot90))
    dataset = dataset.shuffle(2000).repeat()
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=_parse_function, num_parallel_calls=4, batch_size=batch_size))
    dataset = dataset.prefetch(2 * batch_size)
    return dataset


