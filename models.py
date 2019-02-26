
import tensorflow as tf
def inference(images, model_name = 'inception_v1'):

def model_fn(features, labels, mode, params):
    features_columns = params['features_columns']
    images = tf.feature_column.input_layer(features=features, features_columns=feature_columns)

    #calculate logits through CNN
    logits = inference(images)

    if mode in (tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL):
        predicted_indices = tf.argmax(input=logits, axis = 1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.argmax(input=labels, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        tf.summary.scalar('cross_entropy', loss)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class' : predicted_indices,
            'probabilities' : probabilities
        }

        export_outputs = {'predictions' : tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate = params.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {'accuracy' : tf.metrics.accuracy(label_indices, predicted_indices)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


