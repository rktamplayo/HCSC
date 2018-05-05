import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

fc_layer = tf.contrib.layers.fully_connected

class Model(object):

    def __init__(self,
                 base_model,
                 vocab_size,
                 target_len,
                 user_vocab_size,
                 prod_vocab_size,
                 user_mean_freq,
                 prod_mean_freq,
                 embedding_size=300,
                 batch_size=32):
        
        self.vocab_size = vocab_size
        self.user_vocab_size = user_vocab_size
        self.prod_vocab_size = prod_vocab_size
        self.user_mean_freq = user_mean_freq
        self.prod_mean_freq = prod_mean_freq
        self.target_len = target_len
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.base_model = base_model
        
        # PLACEHOLDERS
        self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, None])
        self.input_len = tf.placeholder(tf.int32, shape=[self.batch_size])
        
        self.users = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.user_counts = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
        
        self.prods = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        self.prod_counts = tf.placeholder(tf.float32, shape=[self.batch_size, 1])

        self.targets = tf.placeholder(tf.float32, shape=[self.batch_size, self.target_len])
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        # EMBEDDINGS
        self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], 
                                initializer=xavier_initializer())
        inputs_embedding = tf.nn.embedding_lookup(self.embedding, self.inputs)

        # BASE ENCODER
        if base_model == 'cnn':
            inputs_embedding_expanded = tf.expand_dims(inputs_embedding, -1)
            
            filter_sizes = [3,5,7]
            num_filters = [100,100,100]
            
            outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filters[i]]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name="b")
                    conv = tf.nn.conv2d(
                        inputs_embedding_expanded,
                        W,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    erase = int((7 - filter_size) / 2)
                    if erase != 0:
                        h = h[:,erase:-erase,:,:]
                    h = tf.nn.dropout(h, self.keep_prob)
                    outputs.append(h)
            
            encode = tf.concat(outputs, axis=3)
            encode = tf.squeeze(encode, 2)
        elif base_model == 'rnn':
            fw_cell = tf.nn.rnn_cell.LSTMCell(embedding_size/2)
            bw_cell = tf.nn.rnn_cell.LSTMCell(embedding_size/2)
        
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
            
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs_embedding[:,3:-3,:], 
                                                     sequence_length=self.input_len, dtype=tf.float32)
            encode = tf.concat(outputs, 2)
        elif base_model == 'hcwe':
            fw_cell = tf.nn.rnn_cell.LSTMCell(embedding_size/4)
            bw_cell = tf.nn.rnn_cell.LSTMCell(embedding_size/4)
        
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
            
            outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs_embedding[:,3:-3,:], 
                                                     sequence_length=self.input_len, dtype=tf.float32)
            encode1 = tf.concat(outputs, 2)
            
            inputs_embedding_expanded = tf.expand_dims(inputs_embedding, -1)
            
            filter_sizes = [3,5,7]
            num_filters = [50,50,50]
            
            outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv_maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filters[i]]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters[i]]), name="b")
                    conv = tf.nn.conv2d(
                        inputs_embedding_expanded,
                        W,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    erase = int((7 - filter_size) / 2)
                    if erase != 0:
                        h = h[:,erase:-erase,:,:]
                    h = tf.nn.dropout(h, self.keep_prob)
                    outputs.append(h)
            
            encode2 = tf.concat(outputs, axis=3)
            encode2 = tf.squeeze(encode2, 2)
            
            encode = tf.concat([encode1, encode2], -1)
        
        W0 = tf.get_variable("W0", [embedding_size, target_len], initializer=xavier_initializer())
        b0 = tf.Variable(tf.constant(0.0, shape=[target_len]))
        add_loss = 0
        
        # ATTENTION POOLING
        self.user_embedding = tf.get_variable("user_embedding", [self.user_vocab_size, embedding_size], 
                                initializer=xavier_initializer())
        self.prod_embedding = tf.get_variable("prod_embedding", [self.prod_vocab_size, embedding_size], 
                                initializer=xavier_initializer())
        
        self.user_embed = tf.nn.embedding_lookup(self.user_embedding, self.users)
        self.prod_embed = tf.nn.embedding_lookup(self.prod_embedding, self.prods)
        
        # 1. SPECIFIC VECTORS
        vu_spec, self.a_ud = self.reduce_spec(encode, self.user_embed, 1, embedding_size, "user")
        vp_spec, self.a_pd = self.reduce_spec(encode, self.prod_embed, 1, embedding_size, "prod")
        
        # 2. SHARED VECTORS              
        mean = tf.reduce_mean(encode, 1)
        
        user_mean = fc_layer(mean, embedding_size, activation_fn=None)
        vu_share_weights = tf.matmul(user_mean, tf.transpose(self.user_embedding, [1,0]))
        vu_share_weights = tf.expand_dims(tf.nn.softmax(vu_share_weights), -1)
        vu_share = tf.multiply(vu_share_weights, self.user_embedding)
        vu_share = tf.expand_dims(tf.reduce_sum(vu_share, 1), -2)
        vu_share, self.a_us = self.reduce_spec(encode, vu_share, 1, embedding_size, "user1")
        
        prod_mean = fc_layer(mean, embedding_size, activation_fn=None)
        vp_share_weights = tf.matmul(prod_mean, tf.transpose(self.user_embedding, [1,0]))
        vp_share_weights = tf.expand_dims(tf.nn.softmax(vp_share_weights), -1)
        vp_share = tf.multiply(vp_share_weights, self.user_embedding)
        vp_share = tf.expand_dims(tf.reduce_sum(vp_share, 1), -2)
        vp_share, self.a_ps = self.reduce_spec(encode, vp_share, 1, embedding_size, "prod1")
        
        # 3. GATE
        freq_u_norm = self.user_counts / self.user_mean_freq
        self.lambda_u = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
        self.LU = tf.reduce_mean(self.lambda_u, -1)
        self.k_u = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
        self.KU = tf.reduce_mean(self.k_u, -1)
        gate_u = 1 - tf.exp(-tf.pow(freq_u_norm / tf.nn.relu(self.lambda_u), tf.nn.relu(self.k_u)))
        self.gu = tf.reduce_mean(gate_u, -1)
        vu = gate_u * vu_spec + (1-gate_u) * vu_share
        
        freq_p_norm = self.prod_counts / self.prod_mean_freq # batch, 1
        self.lambda_p = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
        self.LP = tf.reduce_mean(self.lambda_p, -1)
        self.k_p = tf.Variable(tf.constant(1.0, shape=[1, embedding_size]))
        self.KP = tf.reduce_mean(self.k_p, -1)
        gate_p = 1 - tf.exp(-tf.pow(freq_p_norm / tf.nn.relu(self.lambda_p), tf.nn.relu(self.k_p)))
        self.gp = tf.reduce_mean(gate_p, -1)
        vp = gate_p * vp_spec + (1-gate_p) * vp_share
        
        Wg = tf.get_variable("Wg", [embedding_size*2, embedding_size], initializer=xavier_initializer())
        bg = tf.Variable(tf.constant(0.0, shape=[embedding_size]))
        h0 = tf.concat([vu, vp], -1)
        gate = tf.sigmoid(tf.nn.xw_plus_b(h0, Wg, bg))
        self.gup = tf.reduce_mean(gate, -1)
        h = gate * vu + (1-gate) * vp
        
        scores_u_spec = tf.nn.xw_plus_b(vu_spec, W0, b0)
        loss_u_spec = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u_spec, labels=self.targets)
        add_loss += tf.reduce_mean(loss_u_spec)
        
        scores_p_spec = tf.nn.xw_plus_b(vp_spec, W0, b0)
        loss_p_spec = tf.nn.softmax_cross_entropy_with_logits(logits=scores_p_spec, labels=self.targets)
        add_loss += tf.reduce_mean(loss_p_spec)
        
        scores_u_share = tf.nn.xw_plus_b(vu_share, W0, b0)
        loss_u_share = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u_share, labels=self.targets)
        add_loss += tf.reduce_mean(loss_u_share)
        
        scores_p_share = tf.nn.xw_plus_b(vp_share, W0, b0)
        loss_p_share = tf.nn.softmax_cross_entropy_with_logits(logits=scores_p_share, labels=self.targets)
        add_loss += tf.reduce_mean(loss_p_share)
        
        scores_u = tf.nn.xw_plus_b(vu, W0, b0)
        loss_u = tf.nn.softmax_cross_entropy_with_logits(logits=scores_u, labels=self.targets)
        add_loss += tf.reduce_mean(loss_u)
        
        scores_p = tf.nn.xw_plus_b(vp, W0, b0)
        loss_p = tf.nn.softmax_cross_entropy_with_logits(logits=scores_p, labels=self.targets)
        add_loss += tf.reduce_mean(loss_p)
        
        scores = tf.nn.xw_plus_b(h, W0, b0)
        
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.targets)
        self.loss = tf.reduce_mean(losses) + add_loss
        
        predictions = tf.argmax(scores, 1)
        self.predictions = predictions
        self.count = tf.cast(tf.equal(predictions, tf.argmax(self.targets, 1)), 'float')
        
        optimizer = tf.train.AdadeltaOptimizer(1.0, 0.95, 1e-6)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        capped_grads_and_vars = []
        for gv in grads_and_vars:
            #print(gv)
            capped_grads_and_vars.append((tf.clip_by_norm(gv[0], clip_norm=3, axes=[0]), gv[1]))
        self.updates = optimizer.apply_gradients(capped_grads_and_vars)
        
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
    
    def reduce_spec(self, x, y, axis, size, name):
        X = tf.get_variable("X" + name, shape=[size, size], initializer=xavier_initializer())
        Y = tf.get_variable("Y" + name, shape=[size, size], initializer=xavier_initializer())
        b = tf.Variable(tf.zeros([size]))
        z = tf.get_variable("z" + name, shape=[size], initializer=xavier_initializer())
        
        sem = tf.tensordot(x, X, 1)
        sem.set_shape(x.shape)
        
        user = tf.tensordot(y, Y, 1)
        
        weights = tf.nn.tanh(sem + user + b)
        
        weights = tf.tensordot(weights, z, 1)
        weights.set_shape(x.shape[:-1])
        
        ret_weights = weights
        weights = tf.nn.softmax(weights)
        weights = tf.expand_dims(weights, -1)
        
        attended = tf.multiply(x, weights)
        attended = tf.reduce_sum(attended, axis)

        return attended, ret_weights
        
    def step(self,
             session,
             inputs,
             targets,
             users,
             prods,
             input_len,
             user_counts, 
             prod_counts,
             training=True):
        
        max_len = np.max([len(x) for x in inputs])
        pad = 3
        inputs = self.add_pad(inputs, max_len, pad)
        
        input_feed = {}
        input_feed[self.inputs] = inputs
        input_feed[self.input_len] = input_len
        input_feed[self.targets] = targets
        input_feed[self.users] = users
        input_feed[self.prods] = prods
        input_feed[self.user_counts] = user_counts
        input_feed[self.prod_counts] = prod_counts
        
        if training:
            input_feed[self.keep_prob] = 0.5
            output_feed = [self.loss, self.updates]
        else:
            input_feed[self.keep_prob] = 1.0
            output_feed = [self.predictions, self.count, self.KU, self.LU, self.KP, self.LP,
                           self.a_ud, self.a_us, self.a_pd, self.a_ps, self.gu, self.gp, self.gup]
        
        outputs = session.run(output_feed, input_feed)
        
        return outputs
    
    def add_pad(self, data, max_len, pad):
        new_data = []
        for instance in data:
            new_instance = [0] * pad + list(instance) + [0] * (max_len - len(instance)) + [0] * pad
            new_data.append(new_instance)
        return np.array(new_data)
