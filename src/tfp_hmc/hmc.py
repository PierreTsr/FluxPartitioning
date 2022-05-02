"""
    hmc.py
    Created by pierre
    4/26/22 - 11:34 AM
    Description:
    # Enter file description
 """

import functools as ft
from math import ceil

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors


def prior_log_prob_fn(prior, param):
    return tf.reduce_sum(prior.log_prob(param))


def bnn_log_prob_fn(model, inputs, target, params, get_mean=False):
    model.set_params(params)
    pred_mean, pred_var = tf.unstack(model(inputs), axis=-1)
    pred_dists = tfd.Normal(loc=pred_mean, scale=tf.sqrt(pred_var))
    if get_mean:
        return tf.reduce_mean(pred_dists.log_prob(target))
    return tf.reduce_sum(pred_dists.log_prob(target))


def tracer_factory(model, inputs, y):
    return lambda params: ft.partial(bnn_log_prob_fn, model, inputs, y, get_mean=True)(params)


def target_log_prob_fn_factory(prior, model, x_train, y_train):
    # This signature is forced by TFP's HMC kernel which calls log_prob_fn(*chains).
    def target_log_prob_fn(*params):
        flat_params = tf.concat([tf.reshape(param, [-1]) for param in params], 0)
        log_prob = prior_log_prob_fn(prior, flat_params)
        log_prob += bnn_log_prob_fn(model, x_train, y_train, params)
        return log_prob

    return target_log_prob_fn


def trace_fn(current_state, kernel_results, summary_freq=10, callbacks=()):
    """Can be passed to the HMC kernel to obtain a trace of intermediate
    kernel results and histograms of the network parameters in Tensorboard.
    """
    step = kernel_results.step
    with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
        return kernel_results, [cb(*current_state) for cb in callbacks]


@tf.function(experimental_compile=True)
def sample_chain(*args, **kwargs):
    """Compile static graph for tfp.mcmc.sample_chain to improve performance.
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)


def sample_chain_sequentially(num_results, sequence_len, current_state, prev_kernel_results, *args, **kwargs):
    num_steps = [sequence_len] * (num_results // sequence_len)
    if num_results % sequence_len > 0:
        num_steps += [num_results % sequence_len]
    chains = []
    with tqdm(total=num_results, position=0) as pbar:
        for n in num_steps:
            chain, trace, final_kernel_results = sample_chain(*args, num_results=n, current_state=current_state, previous_kernel_results=prev_kernel_results, **kwargs)
            prev_kernel_results = final_kernel_results
            current_state = tf.nest.map_structure(lambda c: c[-1], chain)
            acceptance = trace[0].inner_results.is_accepted.numpy().mean()
            pbar.set_postfix_str("Acceptance rate: {r:.2f}".format(r=acceptance))
            pbar.update(n)
            chains.append([param.numpy() for param in chain])
    chains = [np.concatenate([chain[i] for chain in chains], axis=0) for i in range(len(chains[0]))]
    return chains, final_kernel_results


def run_hmc(
        target_log_prob_fn,
        step_size=1e-2,
        num_leapfrog_steps=10,
        num_burnin_steps=1000,
        num_results=1000,
        seq_len=10,
        current_state=None,
        resume=None,
        sampler="hmc",
        log_dir="../etc/logs/hmc/",
        step_size_adapter="dual_averaging",
        **kwargs,
):
    """Use adaptive HMC to generate a Markov chain of length num_results.

    Args:
        target_log_prob_fn {callable}: Determines the stationary distribution
        the Markov chain should converge to.

    Returns:
        burnin(s): Discarded samples generated during warm-up
        chain(s): Markov chain(s) of samples distributed according to
            target_log_prob_fn (if converged)
        trace: the data collected by trace_fn
        final_kernel_result: kernel results of the last step (in case the
            computation needs to be resumed)
    """
    err = "Either current_state or resume is required when calling run_hmc"
    assert current_state is not None or resume is not None, err

    summary_writer = tf.summary.create_file_writer(str(log_dir))

    step_size_adapter = {
        "simple": tfp.mcmc.SimpleStepSizeAdaptation,
        "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
    }[step_size_adapter]
    if sampler == "nuts":
        kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size)
        adaptive_kernel = step_size_adapter(
            kernel,
            num_adaptation_steps=num_burnin_steps,
            step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                step_size=new_step_size
            ),
            step_size_getter_fn=lambda pkr: pkr.step_size,
            log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        )
    elif sampler == "hmc":
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )
        adaptive_kernel = step_size_adapter(
            kernel, num_adaptation_steps=num_burnin_steps
        )

    prev_kernel_results = adaptive_kernel.bootstrap_results(current_state)
    step = 0

    tf.summary.trace_on(graph=True, profiler=False)

    # chain, trace, _ = sample_chain(
    #     kernel=adaptive_kernel,
    #     current_state=current_state,
    #     previous_kernel_results=prev_kernel_results,
    #     num_results=num_burnin_steps + num_results,
    #     trace_fn=ft.partial(trace_fn, summary_freq=20),
    #     return_final_kernel_results=True,
    #     **kwargs,
    # )
    chain, kernel_results = sample_chain_sequentially(num_burnin_steps + num_results, seq_len, current_state, prev_kernel_results, kernel=adaptive_kernel, trace_fn=ft.partial(trace_fn, summary_freq=20), return_final_kernel_results=True, **kwargs)

    with summary_writer.as_default():
        tf.summary.trace_export(name="hmc_trace", step=step)
    summary_writer.close()

    burnin, samples = zip(*[(t[:-num_results], t[-num_results:]) for t in chain])
    return burnin, samples


def predict_from_chain(chain, model, x_test, uncertainty="aleatoric+epistemic", n_samples=1000):
    """Takes a Markov chain of NN configurations and does the actual
    prediction on a test set X_test including aleatoric and optionally
    epistemic uncertainty estimation.
    """
    err = f"unrecognized uncertainty type: {uncertainty}"
    assert uncertainty in ["aleatoric", "aleatoric+epistemic"], err

    if uncertainty == "aleatoric":
        post_params = [tf.reduce_mean(t, axis=0) for t in chain]
        model.set_params(post_params)
        y_mean, y_var = tf.unstack(model(x_test), axis=-1)
        return y_mean, y_var

    if uncertainty == "aleatoric+epistemic":
        restructured_chain = [
            [tensor[i] for tensor in chain] for i in range(len(chain[0]))
        ]

        @tf.function
        def predict(params):
            model.set_params(params)
            pred = model(x_test, partitioning=True)
            return pred

        preds = [predict(params) for params in restructured_chain]
        restructured_preds = [[pred[i] for pred in preds] for i in range(len(preds[0]))]
        out = []
        samples = np.random.choice(len(preds), (n_samples,), replace=False)
        for preds in restructured_preds:
            preds = tf.gather(preds, samples, axis=0)
            y_mean_mc_samples, y_var_mc_samples = tf.unstack(preds, axis=-1)
            y_mean, y_var_epist = tf.nn.moments(y_mean_mc_samples, axes=0)
            y_var_aleat = tf.reduce_mean(y_var_mc_samples, axis=0)
            y_var_tot = y_var_epist + y_var_aleat
            out.append(tf.stack((y_mean, y_var_tot), axis=-1))
        return out


def get_map_trace(target_log_prob_fn, state, n_iter=1000, save_every=10, callbacks=()):
    optimizer = tf.optimizers.Adam(1e-3)

    @tf.function
    def minimize():
        optimizer.minimize(lambda: -target_log_prob_fn(*state), state)

    state_trace, cb_trace = [], [[] for _ in callbacks]
    for i in tqdm(range(n_iter)):
        if i % save_every == 0:
            state_trace.append(state)
            for trace, cb in zip(cb_trace, callbacks):
                trace.append(cb(state).numpy())
        minimize()

    return state_trace, cb_trace


def get_best_map_state(map_trace, map_log_probs):
    # map_log_probs[0/1]: train/test log probability
    test_set_max_log_prob_idx = np.argmax(map_log_probs[1])
    # Return MAP params that achieved highest test set likelihood.
    return map_trace[test_set_max_log_prob_idx]


def nest_concat(*args, axis=0):
    """Utility function for concatenating a new Markov chain or trace with
    older ones when resuming a previous calculation.
    """
    return tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)
