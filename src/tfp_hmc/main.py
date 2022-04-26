"""
    test.py
    Created by pierre
    4/22/22 - 11:30 AM
    Description:
    # Enter file description
 """
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow import keras

from flux.flux_preprocessing import load_dataset
from flux.flux_viz import quad_viz
from tfp_hmc.hmc import tracer_factory, get_map_trace, target_log_prob_fn_factory, get_best_map_state, run_hmc, \
    predict_from_chain
from tfp_hmc.model import FluxModel

tfd = tfp.distributions


def main(argv=None):
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)

    print("Loading data...")
    train, test, val, EV1_train1, EV2_train1, NEE_train1, label_train, EV1_test1, EV2_test1, NEE_test1, label_test, \
    EV1_val1, EV2_val1, NEE_val1, label_val, NEE_max_abs = load_dataset('NNinput_SCOPE_US_Ha1_1314.csv')

    n_obs_train = NEE_train1.shape[0]
    n_obs_test = NEE_test1.shape[0]
    n_obs_val = NEE_val1.shape[0]
    n_iter = 2000
    burn_in = 1000
    n_samples = 100
    leapfrog_steps = 200
    param_prior = tfd.Normal(0, 0.2)

    print("Building model...")
    model = FluxModel(EV1_train1.shape[1], EV2_train1.shape[1])
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), )
    model({'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1})

    log_prob_tracers = (
        tracer_factory(model, {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
                       NEE_train1),
        tracer_factory(model, {'APAR_input': label_test, 'EV_input1': EV1_test1, 'EV_input2': EV2_test1}, NEE_test1),
    )

    init_state = model.trainable_weights

    print("Searching MAP estimate...")
    trace, log_probs = get_map_trace(
        target_log_prob_fn_factory(param_prior, model,
                                   {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
                                   NEE_train1),
        init_state,
        n_iter=3000,
        callbacks=log_prob_tracers,
    )
    best_map_params = get_best_map_state(trace, log_probs)
    model.set_params(best_map_params)
    nee_map_train, gpp_map_train, reco_map_train = model(
        {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
        partitioning=True
    )
    nee_map_test, gpp_map_test, reco_map_test = model(
        {'APAR_input': label_test, 'EV_input1': EV1_test1, 'EV_input2': EV2_test1},
        partitioning=True
    )

    print("Running HMC...")
    target_log_prob_fn = target_log_prob_fn_factory(param_prior, model,
                                                    {'APAR_input': label_train, 'EV_input1': EV1_train1,
                                                     'EV_input2': EV2_train1}, NEE_train1)
    _, samples1, _, _ = run_hmc(
        target_log_prob_fn,
        num_leapfrog_steps=leapfrog_steps,
        current_state=best_map_params,
        num_results=n_iter,
        num_burnin_steps=burn_in
    )

    print("Computing predictions...")
    nee_dist, gpp_dist, reco_dist = predict_from_chain(samples1, model,
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

    nee_dist, gpp_dist, reco_dist = predict_from_chain(samples1, model,
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

    fig, ax = quad_viz(train, test, "NEE")
    plt.show()
    fig, ax = quad_viz(train, test, "GPP")
    plt.show()
    fig, ax = quad_viz(train, test, "Reco")
    plt.show()

    _, samples2, _, _ = run_hmc(
        target_log_prob_fn,
        num_leapfrog_steps=leapfrog_steps,
        current_state=best_map_params,
        num_results=n_iter,
        num_burnin_steps=burn_in
    )

    parameters = tf.concat([tf.reshape(param, (n_iter, -1)) for param in samples1], 1)
    parameters2 = tf.concat([tf.reshape(param, (n_iter, -1)) for param in samples2], 1)
    tmp = np.stack([parameters, parameters2], axis=2).astype(np.float64)
    for i in range(4):
        random_param = np.random.randint(0, parameters.shape[1])
        np.save(Path("../etc/diagnostic/flux_nn") / ("parameter" + str(i) + ".npy"), tmp[:, random_param, :])
    np.save(Path("../etc/diagnostic/flux_nn") / "full_parameters.npy", tmp)


if __name__ == "__main__":
    raise SystemExit(main())
