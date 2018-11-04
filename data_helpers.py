import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def batch_iter(data, batch_size, num_epochs, shuffle = True):
    """
    Generate batches
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]

def read_source_data():
    data = pd.read_pickle("./demo/small_data.pkl")
    sparse_features = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
    target = ['rating']

    # transfer data into encoding-part
    for features in sparse_features:
        lbl = LabelEncoder()
        data[features] = lbl.fit_transform(data[features])

    # count number of unique features for each sparse field
    sparse_features_dim = {feat: data[feat].nunique() for feat in sparse_features}

    # generate input data, and label
    model_input = np.array([data[feat].value for feat in sparse_features])
    model_label = np.array(data[target])
    for i in range(model_label.shape[0]):
        model_label[i] = 1 if model_label[i] > 2 else 0

    return model_input, model_label, {"sparse": sparse_features_dim, "dense": []}

def tf_weight_sigmoid_ce_with_logits(label=None, logits=None, sample_weight=None):
    return tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=label, logits=logits), sample_weight)