import tensorflow as tf
from tensorflow.keras.models import Model


def layer_output_noSIF(model_NEE, layer_name, label_input, test_input1, test_input2):
    layer_model = Model(inputs=model_NEE.input,
                        outputs=model_NEE.get_layer(layer_name).output)
    inter_output = layer_model.predict({'APAR_input': label_input,
                                        'EV_input1': test_input1,
                                        'EV_input2': test_input2})
    return inter_output


def fluxes_SIF_predict_noSIF(model_NEE, label, EV1, EV2, NEE_max_abs):
    NEE_NN = (layer_output_noSIF(model_NEE, 'NEE', label, EV1, EV2) * NEE_max_abs)
    NEE_NN = tf.reshape(NEE_NN, (NEE_NN.shape[0],))

    GPP_NN = (layer_output_noSIF(model_NEE, 'GPP', label, EV1, EV2) * NEE_max_abs)
    GPP_NN = tf.reshape(GPP_NN, (NEE_NN.shape[0],))

    Reco_NN = (layer_output_noSIF(model_NEE, 'Reco', label, EV1, EV2) * NEE_max_abs)
    Reco_NN = tf.reshape(Reco_NN, (NEE_NN.shape[0],))

    return NEE_NN, GPP_NN, Reco_NN


def get_layer_model(model, name):
    layer_model = Model(inputs=model.input,
                        outputs=model.get_layer(name).output)
    layer_model.compile()
    return layer_model
