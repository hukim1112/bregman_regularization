import tensorflow as tf
from datasets import datasets
from models import vanilla, bregman

def main():
    params = {'learning_rate' : 0.001,
     			'batch_size' : 64,
     			'pretrained_model' : '/home/dan/prj/checkpoints/inception_v1/inception_v1.ckpt',
     			 'model_dir' : '/home/dan/prj/checkpoints/flowers/vanilla/model',
     			 'iteration' : 20000,
     			  'num_classes' : 5,
     			   'train_datadir' : '/home/dan/prj/datasets/flowers/flower_example1/train',
     			    'eval_datadir' : '/home/dan/prj/datasets/flowers/flower_example1/test'}

    classifier = vanilla.model(params)
    classifier.train(params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
