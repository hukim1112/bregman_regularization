import tensorflow as tf
import os
import sys
# add package path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from preprocessing import inception_preprocessing
import random
import os
import shutil
from . import flower_dataset

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
    flower_category = set(
        ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips'])
    if flower_category.issubset(sub_dir):
        return True
    else:
        return False


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
            os.makedirs(os.path.join(
                dest_dir, split_name, category), exist_ok=True)

    # Copy src_file in original directory to dest separated by splits("train", "eval", "test")
    for split_name in splited_list.keys():
        path = os.path.join(dest_dir, split_name)
        for src_file in splited_list[split_name]:
            filename = os.path.basename(src_file)
            category = os.path.split(os.path.dirname(src_file))[1]
            dest_file = os.path.join(path, category, filename)
            shutil.copyfile(src_file, dest_file)
    return


def split_list(_list, amount):
    return _list[:amount], _list[amount:]


def get_train_split(dataset_dir, dest_dir, train_dataset_split_num):
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
    for num in train_dataset_split_num:
        # Make directories of splited train dataset
        _path = os.path.join(dest_dir, 'train_' + str(num))
        for category in class_names_to_ids.keys():
            os.makedirs(os.path.join(_path, category), exist_ok=True)
        splited_list, _ = split_list(filepaths, num)
        print('ss')
        for src_file in splited_list:
            filename = os.path.basename(src_file)
            category = os.path.split(os.path.dirname(src_file))[1]
            dest_file = os.path.join(_path, category, filename)
            shutil.copyfile(src_file, dest_file)
    return


def main():
    dict_of_split_info = {'train': 2500, 'eval': 670, 'test' : 500}
    #dict_of_split_info = {'train': 2500, 'test': 1170}
    splits_name = ""
    for i in dict_of_split_info.keys():
        splits_name = splits_name + " " + i

    print("split flower dataset into {}".format(splits_name))
    dataset_dir = "/home/dan/prj/datasets/flowers/flower_photos"
    dest_dir = '/home/dan/prj/datasets/flowers/flower_example1'
    get_split_dataset(dataset_dir, dest_dir, dict_of_split_info)


if __name__ == '__main__':
    main()
