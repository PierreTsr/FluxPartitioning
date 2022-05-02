"""
    model.py
    Created by pierre
    4/26/22 - 11:33 AM
    Description:
    # Enter file description
 """
import tensorflow as tf
from keras.layers import Dense, Layer
from keras.models import Model


class GPPLayer(Layer):
    def __init__(self, hidden_dim=32, activation="relu"):
        super().__init__()
        self.dense_gpp1 = Dense(hidden_dim, activation=activation, name="hidden1_1")
        self.dense_gpp2 = Dense(2, activation=None, name="ln_GPP")

    def call(self, inputs, *args, **kwargs):
        apar, ev_inp = inputs
        x = self.dense_gpp1(ev_inp)
        x = self.dense_gpp2(x)
        x_mean, x_log_var = tf.unstack(x, axis=-1)
        x_var = tf.exp(x_log_var) * apar
        x_mean = tf.exp(x_mean) * apar
        return tf.stack((x_mean, x_var), axis=-1)


class RecoLayer(Layer):
    def __init__(self, hidden_dim=32, activation="relu"):
        super().__init__()
        self.dense_reco1 = Dense(hidden_dim, activation=activation, name="hidden2_1")
        self.dense_reco2 = Dense(hidden_dim, activation=activation, name="hidden2_2")
        self.dense_reco3 = Dense(2, activation=None, name="ln_Reco")

    def call(self, inputs, *args, **kwargs):
        x = self.dense_reco1(inputs)
        x = self.dense_reco2(x)
        x = self.dense_reco3(x)
        x_mean, x_log_var = tf.unstack(x, axis=-1)
        x_var = tf.exp(x_log_var)
        x_mean = tf.exp(x_mean)
        return tf.stack((x_mean, x_var), axis=-1)


class FluxModel(Model):
    def __init__(self, hidden_dim=32, activation="relu"):
        super().__init__()
        self.gpp_layer = GPPLayer(hidden_dim, activation)
        self.reco_layer = RecoLayer(hidden_dim, activation)
        self.param_shapes = [var.shape for var in self.trainable_weights]

    def set_params(self, params):
        """
        Set the parameters of the model from a unique one-dimensional tensor.

        :param params: parameter values, as a one-dimensional tensor.
        :type params: tf.Tensor
        :return: None
        :rtype: None
        """
        for var, param in zip(self.trainable_weights, params):
            var.assign(param)

    def call(self, inputs, partitioning=False):
        gpp_dist = self.gpp_layer((inputs["APAR_input"], inputs["EV_input1"]))
        gpp_mean, gpp_var = tf.unstack(gpp_dist, axis=-1)
        reco_dist = self.reco_layer(inputs["EV_input2"])
        reco_mean, reco_var = tf.unstack(reco_dist, axis=-1)
        nee_mean = reco_mean - gpp_mean
        nee_var = gpp_var + reco_var
        nee_dist = tf.stack((nee_mean, nee_var), axis=-1)

        if partitioning:
            return nee_dist, tf.stack((gpp_mean, nee_var), axis=-1), tf.stack((reco_mean, nee_var), axis=-1)
        return nee_dist
