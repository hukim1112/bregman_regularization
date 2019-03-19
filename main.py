import tensorflow as tf
from datasets import datasets
from models import vanilla, bregman

def main():
    params = {'learning_rate' : 0.001,
     			'batch_size' : 64,
     			'pretrained_model' : '/home/dan/prj/checkpoints/inception_v1/inception_v1.ckpt',
     			 'model_dir' : '/home/dan/prj/checkpoints/flowers/bregman_116/model',
     			 'iteration' : 20000,
     			  'num_classes' : 5,
     			   'train_datadir' : '/home/dan/prj/datasets/flowers/flower_example1/train_116',
     			    'eval_datadir' : '/home/dan/prj/datasets/flowers/flower_example1/eval'}

    trainer = vanilla.model(params)
    trainer.train(params)
    del trainer
    params['pretrained_model'] = None
    params['eval_datadir'] = '/home/dan/prj/datasets/flowers/flower_example1/test'
    tester = vanilla.model(params)
    score = tester.eval(params)

    print("This model's Final score is {}".format(score))



    # classifier = bregman.model(params)
    # classifier.train(params)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
