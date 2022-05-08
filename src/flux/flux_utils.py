"""
    Description:
    Pre-processing functions for flux partitioning data.
    Authors: Pierre Tessier
 """
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model


def layer_output_noSIF(model_NEE, layer_name, label_input, test_input1, test_input2):
    """
    Compute the output of a given layer in an NN model.

    :param model_NEE: full model to use
    :type model_NEE: keras.Model
    :param layer_name: name of layer whose output is desired
    :type layer_name: str
    :param label_input: input of the model part 1 (APAR)
    :type label_input: tf.Tensor
    :param test_input1: input of the model part 2 (GPP_input)
    :type test_input1: tf.Tensor
    :param test_input2: input of the model part 3 (Reco_input)
    :type test_input2: tf.Tensor
    :return: output of model_NEE[layer_name]
    :rtype: tf.Tensor
    """
    layer_model = Model(inputs=model_NEE.input,
                        outputs=model_NEE.get_layer(layer_name).output)
    inter_output = layer_model.predict({'APAR_input': label_input,
                                        'EV_input1': test_input1,
                                        'EV_input2': test_input2})
    return inter_output


def fluxes_SIF_predict_noSIF(model_NEE, label, EV1, EV2, NEE_max_abs):
    """
    Predict the flux partitioning from a trained NEE model.

    :param model_NEE: full model trained on NEE
    :type model_NEE: keras.Model
    :param label: input of the model part 1 (APAR)
    :type label: tf.Tensor
    :param EV1: input of the model part 2 (GPP_input)
    :type EV1: tf.Tensor
    :param EV2: input of the model part 3 (Reco_input)
    :type EV2: tf.Tensor
    :param NEE_max_abs: normalization factor of NEE
    :type NEE_max_abs: tf.Tensor | float
    :return: corresponding NEE, GPP and Reco value for the provided data
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
    """
    NEE_NN = (layer_output_noSIF(model_NEE, 'NEE', label, EV1, EV2) * NEE_max_abs)
    NEE_NN = tf.reshape(NEE_NN, (NEE_NN.shape[0],))

    GPP_NN = (layer_output_noSIF(model_NEE, 'GPP', label, EV1, EV2) * NEE_max_abs)
    GPP_NN = tf.reshape(GPP_NN, (NEE_NN.shape[0],))

    Reco_NN = (layer_output_noSIF(model_NEE, 'Reco', label, EV1, EV2) * NEE_max_abs)
    Reco_NN = tf.reshape(Reco_NN, (NEE_NN.shape[0],))

    return NEE_NN, GPP_NN, Reco_NN


def get_layer_model(model, name):
    """
    Compile a model to predict a layer output based on a complete model.

    :param model: full model (in flux partitioning, NEE model)
    :type model: keras.Model
    :param name: name of the layer to predict
    :type name: str
    :return: a compiled model taking the same input as `model` and returning `model[name]`
    :rtype: keras.Model
    """
    layer_model = Model(inputs=model.input,
                        outputs=model.get_layer(name).output)
    layer_model.compile()
    return layer_model


def count_out_distribution(df, key, reference="canopy", n_sigma=2):
    """
    Count the number of real values out of the uncertainty band.

    The provided dataframe needs to contain the keys "{key}_{reference}", "{key}_mean" and "{key}_sigma".

    :param df: data frame containing all relevant data
    :type df: pd.DataFrame
    :param key: name of variable to use: "GPP" | "NEE" | "Reco"
    :type key: str
    :param reference: postfix of the target value
    :type reference: str
    :param n_sigma: width of the uncertainty band to consider
    :type n_sigma: float
    :return: number of target values out of mean ± n_sigma * sigma
    :rtype: int
    """
    ref = key + "_" + reference
    mean = key + "_mean"
    sigma = key + "_sigma"

    lower = df[mean] - n_sigma * df[sigma]
    upper = df[mean] + n_sigma * df[sigma]

    above = df[ref] > upper
    below = df[ref] < lower
    return sum(above) + sum(below)
