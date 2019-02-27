
import tensorflow as tf
from backbone import inception
slim = tf.contrib.slim

#param : features_columns, num_classes, learning_rate


def inference(images, params, model_name = 'inception_v1'):
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, endpoints = inception.inception_v1(images, num_classes=params['num_classes'], is_training=True)
    return logits
def model_fn(features, labels, mode, params):
    _features = {'images' : features}
    images = tf.feature_column.input_layer(features=_features, feature_columns=params['feature_columns'])
    images = tf.reshape(
    images, shape=(-1, params['IMAGE_HEIGHT'], params['IMAGE_WIDTH'], params['IMAGE_DEPTH']))
    #calculate logits through CNN
    logits = inference(images, params)
    one_hot_labels = slim.one_hot_encoding(labels, num_classes = params.num_classes)

    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        predicted_indices = tf.argmax(input=logits, axis = 1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = labels
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        tf.summary.scalar('cross_entropy', loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class' : predicted_indices,
            'probabilities' : probabilities
        }

        export_outputs = {'predictions' : tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy' : tf.metrics.accuracy(label_indices, predicted_indices)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


