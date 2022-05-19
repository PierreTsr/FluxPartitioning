"""
    test.py
    Created by pierre
    4/22/22 - 11:30 AM
    Description:
    Flux partitioning with uncertainty estimation using HMC or NUTS.
 """
import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow import keras

from flux.flux_preprocessing import load_dataset
from flux.flux_viz import quad_viz, dual_viz_val
from tfp_hmc.hmc import get_map_trace, target_log_prob_fn_factory, get_best_map_state, run_hmc, \
    predict_from_chain
from tfp_hmc.model import FluxModel

tfd = tfp.distributions

script = True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Runs 4 similar HMC chains for flux partitioning and produces all the experiment visualizations and diagnostics.")
    parser.add_argument("-n", type=int, default=10000, help="number of samples to produce")
    parser.add_argument("-b", type=int, default=1000, help="number of burn-in steps before sampling")
    parser.add_argument("-L", type=int, default=100, help="number of leapfrog steps")
    parser.add_argument("-e", type=float, default=1e-4, help="epsilon - step size for leapfrog integration")
    parser.add_argument("-s", type=str, choices=("hmc", "nuts"), default="nuts", help="type of sampler")
    parser.add_argument("--hidden", type=int, default=32, help="hidden layer dimension")
    parser.add_argument("--sub", type=int, default=1, help="sub-sampling period")

    args = parser.parse_args(argv)

    print("Loading data...")
    train, test, val, EV1_train1, EV2_train1, NEE_train1, label_train, EV1_test1, EV2_test1, NEE_test1, label_test, \
    EV1_val1, EV2_val1, NEE_val1, label_val, NEE_max_abs = load_dataset('NNinput_SCOPE_US_Ha1_1314.csv')

    # ====================== PARAMETERS DEFINITION ======================
    hidden_dim = 12
    map_train_steps = 10000
    n_iter = 5000
    burn_in = 1000
    leapfrog_steps = 1000
    seq_len = 10
    step_size = 3e-4
    sampler = "nuts"
    sample_step = 400
    step_size_adapter = "none"
    param_std = 1
    parallel_iterations = 100

    # Override hard-coded value with script arguments
    if script:
        hidden_dim = args.hidden
        map_train_steps = 10000
        n_iter = args.n
        burn_in = args.b
        leapfrog_steps = args.L
        seq_len = 100
        step_size = args.e
        sampler = args.s
        sample_step = args.sub
        step_size_adapter = "none"
        param_std = 0.5
    # ====================== END PARAMETERS DEFINITION ======================

    param_prior = tfd.Normal(0, param_std)
    # param_prior = tfd.Cauchy(0, 0.2)
    n_samples = min(2000, n_iter)

    # ======================
    print("Building model...")
    model = FluxModel(hidden_dim=hidden_dim)
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3), )
    model({'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1})
    print(model.summary())

    log_prob_tracers = (
        target_log_prob_fn_factory(param_prior, model,
                                   {'APAR_input': label_train, 'EV_input1': EV1_train1, 'EV_input2': EV2_train1},
                                   NEE_train1),
        target_log_prob_fn_factory(param_prior, model,
                                   {'APAR_input': label_test, 'EV_input1': EV1_test1, 'EV_input2': EV2_test1},
                                   NEE_test1),
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
    nee_map_val, gpp_map_val, reco_map_val = model(
        {'APAR_input': label_val, 'EV_input1': EV1_val1, 'EV_input2': EV2_val1},
        partitioning=True
    )

    # ======================
    print("Running HMC...")
    target_log_prob_fn = target_log_prob_fn_factory(param_prior, model,
                                                    {'APAR_input': label_train, 'EV_input1': EV1_train1,
                                                     'EV_input2': EV2_train1}, NEE_train1)

    if sampler == "hmc":
        experiment_dir = Path("../etc/tfp") / (
            f"{sampler}_{step_size_adapter}_n-{n_iter*sample_step}_burn-in-{burn_in}_eps-{step_size}_L-{leapfrog_steps}_sub-{sample_step}"
        ) / (
                             f"prior-{param_std}_hidden-{hidden_dim}"
                         )
    else:
        experiment_dir = Path("../etc/tfp") / (
            f"{sampler}_{step_size_adapter}_n-{n_iter*sample_step}_burn-in-{burn_in}_eps-{step_size}_sub-{sample_step}"
        ) / (
                             f"prior-{param_std}_hidden-{hidden_dim}"
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
        num_steps_between_results=sample_step,
        parallel_iterations=parallel_iterations,
    )

    # plot parameters spread
    parameters = np.concatenate([np.reshape(param.numpy(), (n_iter, -1)) for param in samples1], 1)
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
    max_lag = int(n_iter / 4)
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

    nee_dist, gpp_dist, reco_dist = predict_from_chain(samples1, model,
                                                       {'APAR_input': label_val, 'EV_input1': EV1_val1,
                                                        'EV_input2': EV2_val1}, n_samples=n_samples)
    val["NEE_mean"] = tf.unstack(nee_dist, axis=-1)[0] * NEE_max_abs
    val["NEE_MAP"] = tf.unstack(nee_map_val, axis=-1)[0] * NEE_max_abs
    val["NEE_sigma"] = tf.sqrt(tf.unstack(nee_dist, axis=-1)[1]) * NEE_max_abs
    val["GPP_mean"] = tf.unstack(gpp_dist, axis=-1)[0] * NEE_max_abs
    val["GPP_MAP"] = tf.unstack(gpp_map_val, axis=-1)[0] * NEE_max_abs
    val["GPP_sigma"] = tf.sqrt(tf.unstack(gpp_dist, axis=-1)[1]) * NEE_max_abs
    val["Reco_mean"] = tf.unstack(reco_dist, axis=-1)[0] * NEE_max_abs
    val["Reco_MAP"] = tf.unstack(reco_map_val, axis=-1)[0] * NEE_max_abs
    val["Reco_sigma"] = tf.sqrt(tf.unstack(reco_dist, axis=-1)[1]) * NEE_max_abs

    train_day = train.loc[train.APAR_label == 1,]
    train_night = train.loc[train.APAR_label == 0,]
    test_day = test.loc[test.APAR_label == 1,]
    test_night = test.loc[test.APAR_label == 0,]

    fig, ax = quad_viz(train, test, "NEE", date_break=datetime(2014, 1, 1),
                       unit="(umol m-2 s-1)", filename=experiment_dir / "nee.png")
    # plt.show()
    fig, ax = quad_viz(train, test, "GPP", date_break=datetime(2014, 1, 1),
                       unit="(umol m-2 s-1)", filename=experiment_dir / "gpp.png")
    # plt.show()
    fig, ax = quad_viz(train, test, "Reco", date_break=datetime(2014, 1, 1),
                       colors="Tsoil", unit="(umol m-2 s-1)",
                       filename=experiment_dir / "reco.png")
    # plt.show()

    fig, ax = dual_viz_val(val, "NEE", date_break=datetime(2014, 1, 1),
                           unit="(umol m-2 s-1)", filename=experiment_dir / "nee_val.png")
    # plt.show()
    fig, ax = dual_viz_val(val, "GPP", date_break=datetime(2014, 1, 1),
                           unit="(umol m-2 s-1)", filename=experiment_dir / "gpp_val.png")
    # plt.show()
    fig, ax = dual_viz_val(val, "Reco", date_break=datetime(2014, 1, 1),
                           colors="Tsoil", unit="(umol m-2 s-1)",
                           filename=experiment_dir / "reco_val.png")
    # plt.show()

    fig, ax = quad_viz(train_day, test_day, "NEE", date_break=datetime(2014, 1, 1),
                       unit="(umol m-2 s-1)",
                       filename=experiment_dir / "nee_day.png", postfix="Daytime")
    fig, ax = quad_viz(train_day, test_day, "GPP", date_break=datetime(2014, 1, 1),
                       unit="(umol m-2 s-1)",
                       filename=experiment_dir / "gpp_day.png", postfix="Daytime")
    fig, ax = quad_viz(train_day, test_day, "Reco", date_break=datetime(2014, 1, 1),
                       colors="Tsoil", unit="(umol m-2 s-1)",
                       filename=experiment_dir / "reco_day.png", postfix="Daytime")

    fig, ax = quad_viz(train_night, test_night, "NEE", date_break=datetime(2014, 1, 1),
                       unit="(umol m-2 s-1)",
                       filename=experiment_dir / "nee_night.png", postfix="Nighttime")
    fig, ax = quad_viz(train_night, test_night, "GPP", date_break=datetime(2014, 1, 1),
                       unit="(umol m-2 s-1)",
                       filename=experiment_dir / "gpp_night.png", postfix="Nighttime")
    fig, ax = quad_viz(train_night, test_night, "Reco", date_break=datetime(2014, 1, 1),
                       colors="Tsoil", unit="(umol m-2 s-1)",
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
        log_dir=log_dir,
        num_steps_between_results=sample_step,
        parallel_iterations=parallel_iterations,
    )

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
        log_dir=log_dir,
        num_steps_between_results=sample_step,
        parallel_iterations=parallel_iterations,
    )

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
        log_dir=log_dir,
        num_steps_between_results=sample_step,
        parallel_iterations=parallel_iterations,
    )

    # ======================
    print("Saving experiment results...")
    best_map_params_flat = np.concatenate([np.reshape(param, (-1,)) for param in best_map_params])
    parameters2 = tf.concat([tf.reshape(param.numpy(), (n_iter, -1)) for param in samples2], 1)
    parameters3 = tf.concat([tf.reshape(param.numpy(), (n_iter, -1)) for param in samples3], 1)
    parameters4 = tf.concat([tf.reshape(param.numpy(), (n_iter, -1)) for param in samples4], 1)
    full_parameters = np.stack([parameters, parameters2, parameters3, parameters4], axis=2)

    np.save(str(Path("../etc/diagnostic/flux_nn") / "full_parameters.npy"), full_parameters)
    np.save(str(experiment_dir / "full_parameters.npy"), full_parameters)
    np.save(str(experiment_dir / "map_parameters.npy"), best_map_params_flat)

    os.system("Rscript -e 'library(rmarkdown); rmarkdown::render(\"diagnostic.rmd\", \"html_document\")'")
    os.rename("diagnostic.html", experiment_dir / "diagnostic.html")


if __name__ == "__main__":
    raise SystemExit(main())
