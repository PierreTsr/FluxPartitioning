"""
    model.py
    Created by Pierre Tessier
    4/26/22 - 11:33 AM
    Description:
    Definition of a split-head neural network for flux partitioning.
 """
import tensorflow as tf
from keras.layers import Dense, Layer
from keras.models import Model


class GPPLayer(Layer):
    def __init__(self, hidden_dim=32, activation="relu"):
        """
        Split-Head model for GPP prediction.

        Use a single hidden-layer which predicts log(GPP) and log(Var(GPP)).

        :param hidden_dim: dimension of the single hidden-layer
        :type hidden_dim: int
        :param activation: activation of the hidden-layer
        :type activation: str
        """
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
        """
        Split-Head model for Reco prediction.

        Use a 2 hidden-layers which predict log(Reco) and log(Var(Reco)).

        :param hidden_dim: dimension of the hidden-layers
        :type hidden_dim: int
        :param activation: activation of the hidden-layers
        :type activation: str
        """
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
        """
        Build a NEE model from a GPPLayer and a RecoLayer.

        :param hidden_dim: hidden dimension of all hidden layers
        :type hidden_dim: int
        :param activation: activation of all hidden layers
        :type activation: str
        """
        super().__init__()
        self.gpp_layer = GPPLayer(hidden_dim, activation)
        self.reco_layer = RecoLayer(hidden_dim, activation)

    def set_params(self, params):
        """
        Set the parameters of the model from a list of shaped Tensors.

        :param params: parameter values, in shapes provided by `get_shapes`.
        :type params: list[tf.Tensor]
        """
        for var, param in zip(self.trainable_weights, params):
            var.assign(param)

    def get_shapes(self):
        """
        Get the list of parameter shapes needed by the model.

        :return: list of shapes
        :rtype: list[tf.TensorShape]
        """
        return [var.shape for var in self.trainable_weights]

    def call(self, inputs, partitioning=False):
        """
        Compute prediction for flux partitioning using the split-head model.

        Can make prediction for NEE only or for (NEE, GPP, Reco).

        :param inputs: dictionary of inputs with keys: "APAR_input", "EV_input1" and "EV_input2"
        :type inputs: dict[str, tf.Tensor]
        :param partitioning: whether to predict only NEE or all variables.
        :type partitioning: bool
        :return: NEE prediction or (NEE, GPP, Reco)
        :rtype: tf.Tensor | (tf.Tensor, tf.Tensor, tf.Tensor)
        """
        gpp_dist = self.gpp_layer((inputs["APAR_input"], inputs["EV_input1"]))
        gpp_mean, gpp_var = tf.unstack(gpp_dist, axis=-1)
        reco_dist = self.reco_layer(inputs["EV_input2"])
        reco_mean, reco_var = tf.unstack(reco_dist, axis=-1)
        nee_mean = reco_mean - gpp_mean
        nee_var = gpp_var + reco_var
        nee_dist = tf.stack((nee_mean, nee_var), axis=-1)

        if partitioning:
            return nee_dist, gpp_dist, reco_dist
        return nee_dist
