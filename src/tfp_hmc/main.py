"""
    test.py
    Created by pierre
    4/22/22 - 11:30 AM
    Description:
    # Enter file description
 """
import os
from datetime import datetime
from math import ceil
from pathlib import Path

import numpy as np
import rpy2.robjects as robjects
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from rpy2.robjects import default_converter
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from tensorflow import keras

from flux.flux_preprocessing import load_dataset
from flux.flux_viz import quad_viz
from tfp_hmc.hmc import tracer_factory, get_map_trace, target_log_prob_fn_factory, get_best_map_state, run_hmc, \
    predict_from_chain
from tfp_hmc.model import FluxModel

tfd = tfp.distributions


def subsample(chain, step):
    sampled_chain = [param[::step, ...] for param in chain]
    return sampled_chain


def main(argv=None):
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args(argv)

    print("Loading data...")
    train, test, val, EV1_train1, EV2_train1, NEE_train1, label_train, EV1_test1, EV2_test1, NEE_test1, label_test, \
    EV1_val1, EV2_val1, NEE_val1, label_val, NEE_max_abs = load_dataset('NNinput_SCOPE_US_Ha1_1314.csv')

    hidden_dim = 32
    map_train_steps = 3000
    n_iter = 100000
    burn_in = 10000
    n_samples = 1000
    leapfrog_steps = 50
    seq_len = 400
    step_size = 2e-4
    sampler = "nuts"
    sample_step = 5
    step_size_adapter = "none"
    param_std = 0.6
    param_prior = tfd.Normal(0, param_std)
    # param_prior = tfd.Cauchy(0, 0.2)
    if sample_step is not None:
        n_subsamples = int(ceil(n_iter / sample_step))
    else:
        n_subsamples = n_iter

    # ======================
    print("Building model...")
    model = FluxModel(hidden_dim=hidden_dim)
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), )
    model({'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1})

    log_prob_tracers = (
        tracer_factory(model, {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
                       NEE_train1),
        tracer_factory(model, {'APAR_input': label_test, 'EV_input1': EV1_test1, 'EV_input2': EV2_test1}, NEE_test1),
    )

    init_state = model.trainable_weights

    # ======================
    print("Searching MAP estimate...")
    trace, log_probs = get_map_trace(
        target_log_prob_fn_factory(param_prior, model,
                                   {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
                                   NEE_train1),
        init_state,
        n_iter=map_train_steps,
        callbacks=log_prob_tracers,
    )
    best_map_params = get_best_map_state(trace, log_probs)

    # ======================
    print("Computing MAP predictions...")
    model.set_params(best_map_params)
    nee_map_train, gpp_map_train, reco_map_train = model(
        {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
        partitioning=True
    )
    nee_map_test, gpp_map_test, reco_map_test = model(
        {'APAR_input': label_test, 'EV_input1': EV1_test1, 'EV_input2': EV2_test1},
        partitioning=True
    )

    # ======================
    print("Running HMC...")
    target_log_prob_fn = target_log_prob_fn_factory(param_prior, model,
                                                    {'APAR_input': label_train, 'EV_input1': EV1_train1,
                                                     'EV_input2': EV2_train1}, NEE_train1)

    if sampler == "hmc":
        experiment_dir = Path("../etc/tfp") / (
            f"{sampler}_{step_size_adapter}_n-{n_iter}_burn-in-{burn_in}_eps-{step_size}_L-{leapfrog_steps}_sub-{sample_step}"
        ) / (
                             f"prior-{param_std}"
                         )
    else:
        experiment_dir = Path("../etc/tfp") / (
            f"{sampler}_{step_size_adapter}_n-{n_iter}_burn-in-{burn_in}_eps-{step_size}_sub-{sample_step}"
        ) / (
                             f"prior-{param_std}"
                         )
    experiment_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path("../etc/tfp/logs") / experiment_dir.stem

    _, samples1, _, _ = run_hmc(
        target_log_prob_fn,
        num_leapfrog_steps=leapfrog_steps,
        current_state=best_map_params,
        num_results=n_iter,
        num_burnin_steps=burn_in,
        step_size=step_size,
        sampler=sampler,
        seq_len=seq_len,
        step_size_adapter=step_size_adapter,
        log_dir=log_dir,
    )
    if sample_step is not None:
        samples1 = subsample(samples1, sample_step)

    # plot parameters spread
    parameters = np.concatenate([np.reshape(param.numpy(), (n_subsamples, -1)) for param in samples1], 1)
    fig, axs = plt.subplots(5, 5, figsize=(20, 15))
    for i in range(axs.shape[0]):
        axs[i, i].hist(parameters[:, i], bins=30, density=True)
        for j in range(i):
            axs[i, j].scatter(parameters[:, j], parameters[:, i], s=1, marker="+", c=np.arange(parameters.shape[0]))
        for j in range(i + 1, axs.shape[0]):
            axs[i, j].hist2d(parameters[:, j], parameters[:, i], bins=30, cmap="Blues")
    plt.tight_layout()
    plt.savefig(experiment_dir / "spread.png")
    # plt.show()

    # plot parameters auto-correlation
    fig, axs = plt.subplots(4, 4, figsize=(12, 10))
    axs = axs.flatten()
    max_lag = int(n_subsamples / 4)
    for k in range(len(axs)):
        i = np.random.randint(0, parameters.shape[1])
        auto_cor = []
        mean = np.mean(parameters[:, i])
        var = np.var(parameters[:, i])
        for lag in range(1, max_lag):
            param = parameters[lag:, i]
            param_lagged = parameters[:-lag, i]
            auto_cor.append(np.mean((param - mean) * (param_lagged - mean)) / var)
        axs[k].plot(np.arange(1, max_lag), auto_cor)
    plt.tight_layout()
    plt.savefig(experiment_dir / "parameters.png")
    # plt.show()

    # ======================
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

    # ======================
    print("Running additional chains for diagnostics...")
    _, samples2, _, _ = run_hmc(
        target_log_prob_fn,
        num_leapfrog_steps=leapfrog_steps,
        current_state=best_map_params,
        num_results=n_iter,
        num_burnin_steps=burn_in,
        step_size=step_size,
        sampler=sampler,
        seq_len=seq_len,
        step_size_adapter=step_size_adapter,
        log_dir=log_dir
    )
    if sample_step is not None:
        samples2 = subsample(samples2, sample_step)

    _, samples3, _, _ = run_hmc(
        target_log_prob_fn,
        num_leapfrog_steps=leapfrog_steps,
        current_state=best_map_params,
        num_results=n_iter,
        num_burnin_steps=burn_in,
        step_size=step_size,
        sampler=sampler,
        seq_len=seq_len,
        step_size_adapter=step_size_adapter,
        log_dir=log_dir
    )
    if sample_step is not None:
        samples3 = subsample(samples3, sample_step)

    _, samples4, _, _ = run_hmc(
        target_log_prob_fn,
        num_leapfrog_steps=leapfrog_steps,
        current_state=best_map_params,
        num_results=n_iter,
        num_burnin_steps=burn_in,
        step_size=step_size,
        sampler=sampler,
        seq_len=seq_len,
        step_size_adapter=step_size_adapter,
        log_dir=log_dir
    )
    if sample_step is not None:
        samples4 = subsample(samples4, sample_step)

    # ======================
    print("Saving experiment results...")
    best_map_params_flat = np.concatenate([np.reshape(param, (-1,)) for param in best_map_params])
    parameters2 = tf.concat([tf.reshape(param.numpy(), (n_subsamples, -1)) for param in samples2], 1)
    parameters3 = tf.concat([tf.reshape(param.numpy(), (n_subsamples, -1)) for param in samples3], 1)
    parameters4 = tf.concat([tf.reshape(param.numpy(), (n_subsamples, -1)) for param in samples4], 1)
    full_parameters = np.stack([parameters, parameters2, parameters3, parameters4], axis=2).astype(np.float64)

    np.save(str(Path("../etc/diagnostic/flux_nn") / "full_parameters.npy"), full_parameters)
    np.save(str(experiment_dir / "full_parameters.npy"), full_parameters)
    np.save(str(experiment_dir / "map_parameters.npy"), best_map_params_flat)

    np_cv_rules = default_converter + numpy2ri.converter
    r_save = robjects.r["saveRDS"]
    with localconverter(np_cv_rules) as cv:
        parameters_r = cv.py2rpy(full_parameters)
        r_save(parameters_r, str(Path("../etc/diagnostic/flux_nn") / "full_parameters.rds"))
        r_save(parameters_r, str(experiment_dir / "full_parameters.rds"))

    os.system("Rscript -e 'library(rmarkdown); rmarkdown::render(\"diagnostic.rmd\", \"html_document\")'")
    os.system("html2pdf diagnostic.html diagnostic.pdf")
    os.remove("diagnostic.html")
    os.rename("diagnostic.pdf", experiment_dir / "diagnostic.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
