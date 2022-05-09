"""
    reload_and_plot.py
    Created by pierre
    5/3/22 - 9:54 AM
    Description:
    # Enter file description
 """
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

from flux.flux_preprocessing import load_dataset
from flux.flux_viz import quad_viz
from tfp_hmc.hmc import predict_from_chain
from tfp_hmc.model import FluxModel


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="experiment directory")
    args = parser.parse_args(argv)

    n_samples = 1000

    experiment_dir = Path(args.d)
    parameters = np.load(str(experiment_dir / "full_parameters.npy")).astype(np.float32)
    parameters = parameters[..., 0]
    try:
        map_parameters_flat = np.load(str(experiment_dir / "map_parameters.npy")).astype(np.float32)
    except:
        map_parameters_flat = np.mean(parameters, axis=0)

    model = FluxModel()
    train, test, val, EV1_train1, EV2_train1, NEE_train1, label_train, EV1_test1, EV2_test1, NEE_test1, label_test, \
    EV1_val1, EV2_val1, NEE_val1, label_val, NEE_max_abs = load_dataset('NNinput_SCOPE_US_Ha1_1314.csv')
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), )
    model({'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1})
    shapes = model.get_shapes()

    idx = 0
    samples = []
    for shape in shapes:
        size = tf.reduce_prod(shape)
        param = tf.reshape(parameters[:, idx:idx + size], (-1, *shape))
        samples.append(param)
        idx += size

    idx = 0
    map_parameters = []
    for shape in shapes:
        size = tf.reduce_prod(shape)
        param = tf.reshape(map_parameters_flat[idx:idx + size], shape)
        map_parameters.append(param)
        idx += size

    model.set_params(map_parameters)
    nee_map_train, gpp_map_train, reco_map_train = model(
        {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
        partitioning=True
    )
    nee_map_test, gpp_map_test, reco_map_test = model(
        {'APAR_input': label_test, 'EV_input1': EV1_test1, 'EV_input2': EV2_test1},
        partitioning=True
    )
    print("Computing predictions...")
    nee_dist, gpp_dist, reco_dist = predict_from_chain(samples, model,
                                                       {'APAR_input': label_train, 'EV_input1': EV1_train1,
                                                        'EV_input2': EV2_train1}, n_samples=n_samples)
    train["NEE_mean"] = tf.unstack(nee_dist, axis=-1)[0] * NEE_max_abs
    train["NEE_MAP"] = tf.unstack(nee_map_train, axis=-1)[0] * NEE_max_abs
    train["NEE_sigma"] = tf.sqrt(tf.unstack(nee_dist, axis=-1)[1]) * NEE_max_abs
    train["GPP_mean"] = tf.unstack(gpp_dist, axis=-1)[0] * NEE_max_abs
    train["GPP_MAP"] = tf.unstack(gpp_map_train, axis=-1)[0] * NEE_max_abs
    train["GPP_sigma"] = tf.sqrt(tf.unstack(gpp_dist, axis=-1)[1]) * NEE_max_abs
    train["Reco_mean"] = tf.unstack(reco_dist, axis=-1)[0] * NEE_max_abs
    train["Reco_MAP"] = tf.unstack(reco_map_train, axis=-1)[0] * NEE_max_abs
    train["Reco_sigma"] = tf.sqrt(tf.unstack(reco_dist, axis=-1)[1]) * NEE_max_abs

    nee_dist, gpp_dist, reco_dist = predict_from_chain(samples, model,
                                                       {'APAR_input': label_test, 'EV_input1': EV1_test1,
                                                        'EV_input2': EV2_test1}, n_samples=n_samples)
    test["NEE_mean"] = tf.unstack(nee_dist, axis=-1)[0] * NEE_max_abs
    test["NEE_MAP"] = tf.unstack(nee_map_test, axis=-1)[0] * NEE_max_abs
    test["NEE_sigma"] = tf.sqrt(tf.unstack(nee_dist, axis=-1)[1]) * NEE_max_abs
    test["GPP_mean"] = tf.unstack(gpp_dist, axis=-1)[0] * NEE_max_abs
    test["GPP_MAP"] = tf.unstack(gpp_map_test, axis=-1)[0] * NEE_max_abs
    test["GPP_sigma"] = tf.sqrt(tf.unstack(gpp_dist, axis=-1)[1]) * NEE_max_abs
    test["Reco_mean"] = tf.unstack(reco_dist, axis=-1)[0] * NEE_max_abs
    test["Reco_MAP"] = tf.unstack(reco_map_test, axis=-1)[0] * NEE_max_abs
    test["Reco_sigma"] = tf.sqrt(tf.unstack(reco_dist, axis=-1)[1]) * NEE_max_abs

    train_day = train.loc[train.APAR_label == 1,]
    train_night = train.loc[train.APAR_label == 0,]
    test_day = test.loc[test.APAR_label == 1,]
    test_night = test.loc[test.APAR_label == 0,]

    fig, ax = quad_viz(train, test, "NEE", date_break=datetime(2014, 1, 1), filename=experiment_dir / "nee.png")
    # plt.show()
    fig, ax = quad_viz(train, test, "GPP", date_break=datetime(2014, 1, 1), filename=experiment_dir / "gpp.png")
    # plt.show()
    fig, ax = quad_viz(train, test, "Reco", date_break=datetime(2014, 1, 1), filename=experiment_dir / "reco.png")
    # plt.show()

    fig, ax = quad_viz(train_day, test_day, "NEE", date_break=datetime(2014, 1, 1),
                       filename=experiment_dir / "nee_day.png", postfix="Daytime")
    fig, ax = quad_viz(train_day, test_day, "GPP", date_break=datetime(2014, 1, 1),
                       filename=experiment_dir / "gpp_day.png", postfix="Daytime")
    fig, ax = quad_viz(train_day, test_day, "Reco", date_break=datetime(2014, 1, 1),
                       filename=experiment_dir / "reco_day.png", postfix="Daytime")

    fig, ax = quad_viz(train_night, test_night, "NEE", date_break=datetime(2014, 1, 1),
                       filename=experiment_dir / "nee_night.png", postfix="Nighttime")
    fig, ax = quad_viz(train_night, test_night, "GPP", date_break=datetime(2014, 1, 1),
                       filename=experiment_dir / "gpp_night.png", postfix="Nighttime")
    fig, ax = quad_viz(train_night, test_night, "Reco", date_break=datetime(2014, 1, 1),
                       filename=experiment_dir / "reco_night.png", postfix="Nighttime")


if __name__ == "__main__":
    raise SystemExit(main())
