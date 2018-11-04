import numpy as np
import tensorflow as tf
from dcn import deep_cross_network
import time
import os
import data_helpers

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_checkpoints", 5, "number of checkpoints to store")
tf.flags.DEFINE_integer("batch_size", 200, "Number of training batch size")
tf.flags.DEFINE_flag("dev_sample_percentage", .1, "percentage of the training size")

FLAGS = tf.flags.FLAGS

def process():
    # load data
    x, y, feature_dim_dict, _ = data_helpers.read_source_data()
    # Randomly shuffle data
    np.random.seed(1024)
    shuffle_indices = np.random.permutation(np.arrange(len(y)))

    x_shuffle = x[shuffle_indices]
    y_shuffle = y[shuffle_indices]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))

    x_train, x_dev = x_shuffle[:dev_sample_index], x_shuffle[dev_sample_index:]
    y_train, y_dev = y_shuffle[:dev_sample_index], y_shuffle[dev_sample_index:]

    return x_train, y_train, x_dev, y_dev, feature_dim_dict


def train(x_train, y_train, x_dev, y_dev, feature_dim_dict, hidden_size):
    with tf.Graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement = FLAGS.allow_soft_placement,
            log_device_placement = FLAGS.log_device_placement
        )
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            dcn = deep_cross_network(
                feature_dim_dict= feature_dim_dict,
                hidden_size= hidden_size
            )

            # Define training procedures
            global_step = tf.Variable(0, name="global_step", trainable=True)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(dcn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            log_loss_summary = tf.summary.scalar("loss", dcn.loss)

            # Train Summaries
            train_summary_op = tf.summary.merge([log_loss_summary,  grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([log_loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            sess.run(tf.global_variables_initializer())

            def compute_sample_weight(labels, class_weight=None, sample_weight = None):
                if class_weight is None and sample_weight is None:
                    return np.ones(len(labels))

                sample_weight = np.array(labels)
                for label, weight in class_weight.items():
                    sample_weight[sample_weight == label] = weight
                return sample_weight


            def train_step(x_batch, y_batch, class_weight = None, sample_weight = None):
                """

                :param x_batch:
                :param y_batch:
                :param class_weight:
                :param sample_weight:
                :return:
                """
                feed_dict = {
                    dcn.input_x: x_batch,
                    dcn.input_y: y_batch,
                    dcn.dropout_keep_prob: 0.5,
                    dcn.sample_weight: compute_sample_weight(y_batch, class_weight, sample_weight)
                }
                _, step, summaries, train_loss, logits = sess.run([train_op, global_step,
                                                                   train_summary_op, dcn.loss, dcn.logit], feed_dict)
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_batch, y_batch, class_weight = None, sample_weight = None, writer=None):
                """

                :param x_batch:
                :param y_batch:
                :param class_weight:
                :param sample_weight:
                :param writer
                :return:
                """
                feed_dict = {
                    dcn.input_x : x_batch,
                    dcn.input_y : y_batch,
                    dcn.dropout_keep_prob : 1.0,
                    dcn.sample_weight : compute_sample_weight(y_batch, class_weight, sample_weight)
                }
                step, summaries, loss = sess.run(
                    [global_step, dev_summary_op, dcn.loss],
                    feed_dict
                )
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs
            )
            # Training loop, for each epoch
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 1000 == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)
                    print("")
                if  current_step % 1000 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("saved model checkpoints to {}\n".format(path))

def main(argv = None):
    x_train, y_train, x_test, y_test, feature_dim_dict = process()
    hidden_size = [32,32]
    train(x_train, y_train, x_test, y_test, feature_dim_dict, hidden_size)


if __name__ == '__main__':
    tf.app.run()



