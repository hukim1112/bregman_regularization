import tensorflow as tf
import os
import sys
# add package path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from preprocessing import inception_preprocessing
import random
import os
import shutil


_DATA_URL = 'http://download.tensorflow.org/example_images/flower_photos.tgz'


def download_and_uncompress_tarball(dataset_dir, tarball_url=_DATA_URL):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    tarball_url: The URL of a tarball file.
  """

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(os.path.join(dataset_dir, 'flower_photos')):
    print('Dataset files already exist. Exiting without re-creating them.')
    return
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def _dataset_exists(dataset_dir):
  sub_dir = os.listdir(dataset_dir)
  flower_category = set(['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
  if flower_category.issubset(sub_dir):
    return True
  else:
    return False


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
    dataset = dataset.repeat(0)
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


def get_split_dataset(dataset_dir, dest_dir, dict_of_split_info):
  # Split flower dataset into seperated directory
  """
      arguments
        1. dataset_dir : directory of original dataset uncompressed
        2. dict_of_split_info : dict of split directory names and number of images each split each split.
            example : {'train' : 2500, 'eval' : 500, 'test' : 670}. note that all files are 3670.
      return
        None but images in original directory would be splited into directories of split names     
  """
  # Load dataset from original flower-image names and shuffle them.
  filepaths, class_names_to_ids = flower_dataset.load_data(dataset_dir)
  random.shuffle(filepaths)

  splited_list = {}
  for split_name in dict_of_split_info.keys():
    splited_list[split_name], filepaths = split_list(
        filepaths, dict_of_split_info[split_name])

  # Check total numbers in dictionary to be matched with real total number of files.
  if len(filepaths) != 0:
    print("total number of images is not correct in your split_num dictionary.")
    return

  # Make directories for each category of each split
  for split_name in dict_of_split_info.keys():
    for category in class_names_to_ids.keys():
      os.makedirs(os.path.join(dest_dir, split_name, category), exist_ok=True)

  # Copy src_file in original directory to dest separated by splits("train", "eval", "test")
  for split_name in splited_list.keys():
    path = os.path.join(dest_dir, split_name)
    for src_file in splited_list[split_name]:
      filename = os.path.basename(src_file)
      category = os.path.split(os.path.dirname(src_file))[1]
      print(category)
      dest_file = os.path.join(path, category, filename)
      shutil.copyfile(src_file, dest_file)

  return


def split_list(_list, amount):
  return _list[:amount], _list[amount:]


def main():
  print("Testing dataset of flower dataset")
  dict_of_split_info = {'train': 2500, 'eval': 500, 'test': 670}
  dataset_dir = "/home/dan/prj/datasets/flower_photos"
  dest_dir = '/home/dan/prj/datasets/flower_exp1'
  get_split_dataset(dataset_dir, dest_dir, dict_of_split_info)


if __name__ == '__main__':
  main()
