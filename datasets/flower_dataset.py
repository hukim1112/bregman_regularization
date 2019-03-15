import tensorflow as tf
import os
import sys
# add package path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from preprocessing import inception_preprocessing
import os
import math


def _get_filenames_and_classes(dataset_dir):
    directories = []
    class_names = []
    for dir_name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(dir_name)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def load_categorical_data(dataset_dir):
    filepaths = {}
    class_names = []
    for dir_name in os.listdir(dataset_dir):
        directory = os.path.join(dataset_dir, dir_name)
        if os.path.isdir(directory):
            filepaths[dir_name] = []
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                filepaths[dir_name].append(path)
            class_names.append(dir_name)
      class_names_to_ids = dict(zip(class_names, range(len(class_names))))
      return filepaths, class_names_to_ids


def load_data(dataset_dir):
    filepaths, class_names = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    return filepaths, class_names_to_ids


def _parse_function(filename, label=None):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # image_resized = tf.image.resize_images(image_decoded, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    images = inception_preprocessing.preprocess_image(
        image_decoded, height=224, width=224, is_training=True)

    return images, label


def input_fn(filepaths, class_names_to_ids, batch_size, num_images, mode="training"):
    """An input function for training """

    # Convert the inputs to a Dataset
    dataset_filepath = tf.data.Dataset.from_tensor_slices(
        tf.cast(filepaths, tf.string))
    dataset_class = tf.data.Dataset.from_tensor_slices(
        [class_names_to_ids[os.path.basename(os.path.dirname(filepath))] for filepath in filepaths])
    dataset = tf.data.Dataset.zip((dataset_filepath, dataset_class))
    dataset = dataset.shuffle(num_images)
    if mode != "training":
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat()
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    return dataset


def predict_input_fn(filepaths, class_names_to_ids, batch_size):
    """An input function for evaluation or prediction"""

    # Convert the inputs to a Dataset.
    num_images = len(filepaths)
    dataset_filepath = tf.data.Dataset.from_tensor_slices(
        tf.cast(filepaths, tf.string))
    dataset_class = tf.data.Dataset.from_tensor_slices(None)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)

    # Return the dataset.
    return dataset

def categorical_input_fn(filepaths, class_names_to_ids, categorical_batch_size, mode="training"):
    data = {}
    for name in class_names_to_ids.keys():
        dataset = tf.data.Dataset.from_tensor_slices(tf.cast(
            filepaths[name], tf.string))
        dataset_class = tf.data.Dataset.from_tensor_slices(
        [class_names_to_ids[os.path.basename(os.path.dirname(filepath))] for filepath in filepaths[name]])

        dataset = tf.data.Dataset.zip((dataset_filepath, dataset_class))
        dataset = dataset.shuffle(len(filepaths[name]))
        dataset = dataset.repeat()
        dataset = dataset.map(_parse_function, num_parallel_calls=2)
        if len(filepaths[name])/2 > categorical_batch_size:
            dataset = dataset.batch(categorical_batch_size)
        else:
            dataset = dataset.batch(math.floor(len(filepaths[name])/2))
        dataset = dataset.prefetch(2 * batch_size)
        iterator = dataset.make_one_shot_iterator()
        data[name] = iterator.get_next()
        stacked_data = tf.stack([x for x in data.values()])
        stacked_data = tf.reshape(stacked_data, shape=(-1, 224, 224, 3))
