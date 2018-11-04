import tensorflow as tf
import numpy as np

class deep_cross_network(object):
    def __init__(self, feature_dim_dict,
                 embedding_size=4, seed=1024, l2_reg_lamda=0.0002,
                 keep_prob=0.5, use_batch_norm=True, init_std=0.001,
                 cross_layer_num=3, hidden_size=[32,32]):

        self.seed = seed
        self.field_dim = len(feature_dim_dict["sparse"])
        self.sample_weight = tf.placeholder(tf.float32, [None,], name="sample_weight")
        self.input_x = tf.placeholder(tf.int32, [None, self.field_dim], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.train_flag = tf.constant(True, dtype=tf.bool)
        self.embedding_size = embedding_size
        self.l2_reg_lamda = tf.constant(l2_reg_lamda, dtype=tf.float32)
        self.init_std = init_std
        self.use_batchnorm = use_batch_norm
        self.feature_dic_dict = feature_dim_dict
        self.feature_dim = len(feature_dim_dict["sparse"])
        self.cross_layer_num = cross_layer_num
        self.hidden_size = hidden_size

        # create variable
        self.b = tf.Variable(tf.constant(0.0), name="bias")
        self.embedding_size = []
        self.total_size = self.field_dim * self.embedding_size
        self.sparse_embeddings = [tf.get_variable(name='embed_cate' + str(i) + '-' + feat,
                                                  initializer=tf.random_normal(
                                                      [self.feature_dim["sparse"][feat],
                                                       min(self.embedding_size,
                                                           6 * pow(self.feature_dim["sparse"][feat], 0.25))],
                                                      stddev=self.init_std)) for i, feat in
                                  enumerate(self.feature_dim["sparse"])]
        self.cross_layer_weight = [
            tf.Variable(tf.random_normal([self.total_size, 1], stddev=self.init_std, seed=self.seed)) for i in
            range(self.cross_layer_num)]
        self.cross_layer_bias = [
            tf.Variable(tf.random_normal([self.total_size, 1], stddev=self.init_std, seed=self.seed)) for i in
            range(self.cross_layer_num)]
        self.weights = self._initialize_weights()

        with tf.name_scope("cross_network"):
            embed_list = [tf.nn.embedding_lookup(self.sparse_embeddings[i], self.X[:, i]) for i in
                          range(self.field_dim)]
            embeds = tf.concat(embed_list, axis=-1)
            self._x_0 = tf.reshape(embeds, (-1, self.total_size, 1))
            x_l = self._x_0
            for l in range(self.cross_layer_num):
                x_l = self.f_cross_l(
                    x_l, self.cross_layer_weight[l], self.cross_layer_bias[l]) + x_l

            cross_network_out = tf.reshape(x_l, (-1, self.total_size))

        with tf.name_scope("deep_network"):
            if len(self.hidden_size) > 0:
                fc_input = tf.reshape(
                    embeds, (-1, self.field_dim * self.embedding_size))

                for l in range(len(self.hidden_size)):
                    if self.use_batchnorm:
                        weight = tf.get_variable(name='deep_weight' + str(l),
                                                 shape=[fc_input.get_shape().as_list()[1], self.hidden_size[l]],
                                                 initializer=tf.random_normal_initializer(stddev=self.init_std,
                                                                                          seed=self.seed))
                    # bias = tf.Variable(0.0,name='bias'+str(l))
                    H = tf.matmul(fc_input, weight)  # ,bias
                    H_hat = tf.layers.batch_normalization(H, training=self.train_flag)
                    fc = tf.nn.relu(H_hat)

                    if l < len(self.hidden_size) - 1:
                        fc = tf.cond(self.train_flag, lambda: inverted_dropout(fc, self.keep_prob), lambda: fc)
                    fc_input = fc
                deep_network_out = fc_input

        with tf.name_scope("combination_output_layer"):
            x_stack = cross_network_out
            if len(self.hidden_size) > 0:
                x_stack = tf.concat([x_stack, deep_network_out], axis=1)

            self.logit = tf.add(tf.matmul(x_stack, self.weights['concat_projection']), self.weights['concat_bias'])

        with tf.name_scope("loss"):
            self.out = tf.nn.sigmoid(self.logit, name="out")
            self.loss = tf.losses.log_loss(self.input_y, self.out)

    def f_cross_l(self, x_l, w_l, b_l):
            dot = tf.matmul(self._x_0, x_l, transpose_b=True)
            return tf.tensordot(dot, w_l, 1) + b_l

    def inverted_dropout(self, fc, keep_pron):
        return tf.divide(tf.nn.dropout(fc, keep_pron), keep_pron)

    def _initialize_weights(self):
        weights = dict()
        # embeddings
        weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.cate_feature_size,self.embedding_size],0.0,0.01),
            name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.random_normal([self.feature_dim,1],0.0,1.0),name='feature_bias')


        #deep layers
        num_layer = len(self.hidden_size)
        glorot = np.sqrt(2.0/(self.total_size + self.hidden_size[0]))

        weights['deep_layer_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(self.total_size,self.hidden_size[0])),dtype=np.float32
        )
        weights['deep_bias_0'] = tf.Variable(
            np.random.normal(loc=0,scale=glorot,size=(1,self.hidden_size[0])),dtype=np.float32
        )

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.hidden_size[i - 1] + self.hidden_size[i]))
            weights["deep_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_size[i - 1], self.hidden_size[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["deep_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_size[i])),
                dtype=np.float32)  # 1 * layer[i]

        for i in range(self.cross_layer_num):

            weights["cross_layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.total_size,1)),
                dtype=np.float32)
            weights["cross_bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.total_size,1)),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer


        input_size = self.total_size + self.hidden_size[-1]

        glorot = np.sqrt(2.0/(input_size + 1))
        weights['concat_projection'] = tf.Variable(np.random.normal(loc=0,scale=glorot,size=(input_size,1)),dtype=np.float32)
        weights['concat_bias'] = tf.Variable(tf.constant(0.01),dtype=np.float32)

        return weights
