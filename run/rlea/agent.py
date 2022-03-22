import pandas as pd
import numpy as np
import tensorflow as tf


class EmbeddingModel():
    def __init__(self, args, scope='embedding_model'):
        self.args = args
        self.dense1 = tf.layers.Dense(
            units=args.hidden_size,
            kernel_initializer=tf.zeros_initializer)
        self.sparse_adj = tf.cast(tf.SparseTensor(
            indices=args.adj[0], values=args.adj[1], dense_shape=args.adj[2]), tf.float32)

    def predict(self, ent_idx, embeds):
        embeds = tf.sparse_tensor_dense_matmul(self.sparse_adj, embeds)
        ent_embed = tf.reshape(tf.nn.embedding_lookup(
            embeds, ent_idx), (-1, self.args.dim))

        output = self.dense1(ent_embed)

        return output


class MIEstimator():
    def __init__(self, args, scope='mutual_information_estimator'):
        self.dense1 = tf.layers.Dense(
            units=args.hidden_size,
            kernel_initializer=tf.zeros_initializer,
            use_bias=False,
        )

    def predict(self, s, t):
        hidden = self.dense1(s)

        output = tf.exp(tf.matmul(hidden, tf.transpose(t)))

        return output


class BaseLayer():

    def __init__(self, args, scope='base_layer'):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, shape=[2, ], name='state')
            self.opponent = tf.placeholder(
                tf.int32, shape=[None, ], name='state')
            self.action = tf.placeholder(dtype=tf.int32, name='action')
            self.target = tf.placeholder(dtype=tf.float32, name='target')
            self.embeds = tf.placeholder(dtype=tf.float32, shape=[
                                         args.kgs.entities_num, args.dim], name='embeds')

            em = EmbeddingModel(args)
            estimator = MIEstimator(args)

            s_embed = em.predict(self.state[0], self.embeds)
            t_embed = em.predict(self.state[1], self.embeds)
            o_embeds = em.predict(self.opponent, self.embeds)
            st_mi = estimator.predict(s_embed, t_embed)
            so_mi = estimator.predict(s_embed, o_embeds)

            hidden = tf.layers.dense(
                tf.concat([s_embed, t_embed], axis=-1),
                8, kernel_initializer=tf.zeros_initializer,
                activation=tf.nn.relu
            )

            features = [hidden, ]
            feature_shape = 8
            if args.MIE:
                feature_shape += 1 
                features += [st_mi/(st_mi+tf.reduce_sum(so_mi)+1e-5)]
            hidden = tf.reshape(
                tf.concat(features, axis=-1), [1, feature_shape])

            action_probs = tf.squeeze(tf.layers.dense(
                hidden, 2,
                kernel_initializer=tf.zeros_initializer
            ))

            self.hidden = hidden
            self.action_probs = tf.nn.softmax(action_probs)


class Policy():

    def __init__(self, args, scope='policy'):
        with tf.variable_scope(scope):

            self.model = BaseLayer(args)

            self.picked_action_prob = tf.gather(
                self.model.action_probs, self.model.action)

            # Loss and train op
            self.loss = -tf.log(self.picked_action_prob) * self.model.target

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=args.policy_lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, env, sess):
        entity_pair = np.array([state.source_ent_idx, state.target_ent_idx])
        feed_dict = {self.model.state: entity_pair}

        feed_dict[self.model.opponent] = state.oppenent_idx
        feed_dict[self.model.embeds] = env.embeds

        return sess.run(self.model.action_probs, feed_dict)

    def update(self, state, target, action, env, sess):
        entity_pair = np.array([state.source_ent_idx, state.target_ent_idx])
        feed_dict = {self.model.state: entity_pair,
                     self.model.target: target, self.model.action: action}

        feed_dict[self.model.opponent] = state.oppenent_idx
        feed_dict[self.model.embeds] = env.embeds

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class Baseline():

    def __init__(self, args, scope='value_estimator'):
        with tf.variable_scope(scope):
            self.model = BaseLayer(args)

            self.value_estimate = tf.squeeze(tf.layers.dense(
                self.model.hidden, 1, kernel_initializer=tf.zeros_initializer))
            self.loss = tf.squared_difference(
                self.value_estimate, self.model.target)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=args.policy_lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())

    def predict(self, state, env, sess):
        entity_pair = np.array([state.source_ent_idx, state.target_ent_idx])
        feed_dict = {self.model.state: entity_pair}

        feed_dict[self.model.opponent] = state.oppenent_idx
        feed_dict[self.model.embeds] = env.embeds
        return sess.run(self.value_estimate, feed_dict)

    def update(self, state, target, env, sess):
        entity_pair = np.array([state.source_ent_idx, state.target_ent_idx])
        feed_dict = {self.model.state: entity_pair, self.model.target: target}

        feed_dict[self.model.opponent] = state.oppenent_idx
        feed_dict[self.model.embeds] = env.embeds

        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss