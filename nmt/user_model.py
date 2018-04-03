"""User model"""
import tensorflow as tf


class UserModel():
    """
    User model predicts user utterance from the previous context
    Implemented as a feed-forward neural network
    """

    def __init__(self,
                 n_layers):
        self.n_layers = n_layers
        self.model = tf.layers.dense()
        # difference
        self.comparison_operator = lambda vec1, vec2: tf.reduce_sum(tf.abs(tf.subtract(vec1, vec2)))
        # cosine distance
        # self.comparison_operator = lambda vec1, vec2: tf.losses.cosine_distance(vec1, vec2)

    # context --- embedding of context
    # sess --- TF session
    # output --- predicted user state
    def predict_state(self, sess, context):
        pass

    # true_state --- true user state
    def compare(self, true_state):
        return self.comparison_operator(self.state, true_state)