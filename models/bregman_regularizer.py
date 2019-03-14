import tensorflow as tf


def get_prototypes_from_embeddings(embeddings, labels, class_num):
    class_ids = list(range(class_num))
    class_ids = tf.reshape(tf.constant(class_ids), shape=(len(class_ids), 1))
    # indice corresponding each class on embeddings
    indice = tf.equal(labels, class_ids)
    prototypes = tf.map_fn(lambda x: get_prototypes_each_class(
        embeddings, x), indice, dtype=embeddings.dtype)
    return prototypes


def get_prototypes_each_class(embeddings, indice_each_class):
    # group_embeddings_by_label
    embeddings_of_the_label = tf.gather(
        embeddings, tf.where(indice_each_class))
    embeddings_of_the_label = tf.squeeze(
        embeddings_of_the_label, axis=1)  # shape (n, 1, d) -> (n, d)
    # get embeddings averaged
    prototype = tf.reduce_mean(
        embeddings_of_the_label, axis=0)  # shape (n, d) -> (d)
    return prototype


def get_bregman_loss_from_embeddings(embeddings, labels, prototypes, class_num):
    class_ids = list(range(class_num))
    class_ids = tf.reshape(tf.constant(class_ids), shape=(len(class_ids), 1))
    # indice corresponding each class on embeddings
    indice = tf.equal(labels, class_ids)

    losses = tf.map_fn(lambda x: get_loss_each_prototype(
        embeddings, x[0], x[1]), (prototypes, indice), dtype=embeddings.dtype)
    return losses


def get_loss_each_prototype(embeddings, prototype, indice_each_class):
    embeddings_of_the_label = tf.gather(
        embeddings, tf.where(indice_each_class))
    embeddings_of_the_label = tf.squeeze(
        embeddings_of_the_label, axis=1)  # shape (n, 1, d) -> (n, d)
    # (p1 - x1)^2 + (p2 - x2)^2 + ... + (pn - xn)^2
    #squared_l2_distance = tf.pow(prototype - embeddings_of_the_label, 2)
    squared_l2_distance = tf.reduce_sum(
        tf.pow(prototype - embeddings_of_the_label, 2))
    return squared_l2_distance

def prototypical_classifier(embeddings, labels, prototypes, class_num):
    class_ids = list(range(class_num))
    class_ids = tf.reshape(tf.constant(class_ids), shape=(len(class_ids), 1))
    # indice corresponding each class on embeddings
    indice = tf.equal(labels, class_ids)

    scores = tf.map_fn(lambda x: distance_from_prototypes(x, prototypes), embeddings, dtype=embeddings.dtype)
    return scores

def distance_from_prototypes(embedding, prototypes):
    distance_vector = prototypes - embedding
    l2_distance_from_prototypes = tf.norm(distance_vector, axis=1)
    return -l2_distance_from_prototypes
