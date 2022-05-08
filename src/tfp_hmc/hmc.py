"""
    hmc.py
    Created by Pierre Tessier
    4/26/22 - 11:34 AM
    Description:
    HMC implementation, mostly from https://janosh.dev/blog/hmc-bnn
 """
import functools as ft
from typing import Any
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

def prior_log_prob_fn(prior, param):
    """
    Compute the log-likelihood of a set of parameters given their prior.

    :param prior: prior distribution of the parameters
    :type prior: tfp.distributions.Distribution
    :param param: parameters values as a one-dimensional tensor
    :type param: tf.Tensor
    :return: log-likelihood value
    :rtype: tf.Tensor
    """
    return tf.reduce_sum(prior.log_prob(param))


def bnn_log_prob_fn(model, inputs, target, params, get_mean=False):
    """
    Compute the log-likelihood of a model's predictions given the target values.

    :param model: model to use (needs to be a split-head model)
    :type model: keras.Model
    :param inputs: model inputs (shape depends on model architecture)
    :type inputs: Any
    :param target: target values corresponding to the inputs
    :type target: tf.Tensor
    :param params: parameters as a list of tensors (shapes must match model.trainable_variables())
    :type params: list[tf.Tensor] | tuple[tf.Tensor]
    :param get_mean: if set to True, computes the averaged log-likelihood instead of the sum - must be set to False
    for training
    :type get_mean: bool
    :return: log-likelihood value
    :rtype: tf.Tensor
    """
    model.set_params(params)
    pred_mean, pred_var = tf.unstack(model(inputs), axis=-1)
    pred_dists = tfd.Normal(loc=pred_mean, scale=tf.sqrt(pred_var))
    if get_mean:
        return tf.reduce_mean(pred_dists.log_prob(target))
    return tf.reduce_sum(pred_dists.log_prob(target))


def tracer_factory(model, inputs, y):
    """
    Create a trace function logging the log-likelihood of a model's predictions.

    :param model: model to use (needs to be a split-head model)
    :type model: keras.Model
    :param inputs: model inputs (shape depends on model architecture)
    :type inputs: Any
    :param y: target values corresponding to the inputs
    :type y: tf.Tensor
    :return: function taking a set of parameters and returning a log-likelihood value
    :rtype: function
    """
    return lambda params: ft.partial(bnn_log_prob_fn, model, inputs, y, get_mean=True)(params)


def target_log_prob_fn_factory(prior, model, x_train, y_train):
    """
    Creates a log-likelihood function following the formalism required by tfp.hmc.

    :param prior: prior distribution of the parameters
    :type prior: tfp.distributions.Distribution
    :param model: model to use (needs to be a split-head model)
    :type model: keras.Model
    :param x_train: training inputs (shape depends on model architecture)
    :type x_train: Any
    :param y_train: training targets
    :type y_train: tf.Tensor
    :return: function to provide to the HMC sampler
    :rtype: function
    """
    def target_log_prob_fn(*params):
        """
        Log-likelihood function as needed by the HMC sampler.

        :param params: parameters as a list of tensors (shapes must match model.trainable_variables())
        :type params: tf.Tensor
        :return: total log-likelihood of the model for given prior and training data
        :rtype: tf.Tensor
        """
        flat_params = tf.concat([tf.reshape(param, [-1]) for param in params], 0)
        log_prob = prior_log_prob_fn(prior, flat_params)
        log_prob += bnn_log_prob_fn(model, x_train, y_train, params)
        return log_prob

    return target_log_prob_fn


def trace_fn(current_state, kernel_results, summary_freq=10, callbacks=()):
    """
    Can be passed to the HMC kernel to obtain a trace of intermediate
    kernel results and histograms of the network parameters in Tensorboard.
    """
    # step = kernel_results.step
    # with tf.summary.record_if(tf.equal(step % summary_freq, 0)):
    return kernel_results, [cb(*current_state) for cb in callbacks]


@tf.function(experimental_compile=True)
def sample_chain(*args, **kwargs):
    """
    Compile static graph for tfp.mcmc.sample_chain to improve performance.
    """
    return tfp.mcmc.sample_chain(*args, **kwargs)


def sample_chain_sequentially(num_results, sequence_len, current_state, prev_kernel_results, *args, **kwargs):
    """
    Sequentially calls sample_chain to avoid memory exhaustion, and make it possible to print HMC progress bar.

    Breaks the total chain in multiple sub-chains of length `sequence_length`, and resume the execution after each.
    Each sub-chain is loaded to the CPU before resuming, which avoids VRAM exhaustion. Each execution updates a global
    progress bar which makes it possible to track HMC's total runtime.

    :param num_results: length of the total chain
    :type num_results: int
    :param sequence_len: maximum length of a sub-chain (defines the update frequency of the progress bar)
    :type sequence_len: int
    :param current_state: initial state of the chain (model parameters usually)
    :type current_state: list[tf.Tensor]
    :param prev_kernel_results: results from a previous execution if the chain is resuming one
    :param args: extra arguments to use in tf.mcmc.sample_chain()
    :param kwargs: extra keyword arguments to use in tf.mcmc.sample_chain()
    :return: results of the entire chain
    :rtype: (list[tf.Tensor], Any, Any)
    """

    num_steps = [sequence_len] * (num_results // sequence_len)
    if num_results % sequence_len > 0:
        num_steps += [num_results % sequence_len]
    total_chain = None
    total_trace = None
    with tqdm(total=num_results, position=0) as pbar:
        for n in num_steps:
            chain, trace, final_kernel_results = sample_chain(*args, num_results=n, current_state=current_state, previous_kernel_results=prev_kernel_results, **kwargs)
            prev_kernel_results = final_kernel_results
            current_state = tf.nest.map_structure(lambda c: c[-1], chain)
            try:
                acceptance = trace[0].inner_results.is_accepted.numpy().mean()
            except AttributeError:
                acceptance = trace[0].is_accepted.numpy().mean()
            pbar.set_postfix_str("acceptance rate: {r:.2f}".format(r=acceptance))
            pbar.update(n)
            if total_chain is None:
                total_trace = trace
                total_chain = chain
            else:
                total_trace = nest_concat(total_trace, trace)
                total_chain = nest_concat(total_chain, chain)
    return total_chain, total_trace, final_kernel_results


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
    """
    Run an HMC or NUTS chain.

    :param target_log_prob_fn: log-likelihood function of the kernel
    :type target_log_prob_fn: function
    :param step_size: HMC or NUTS step-size (epsilon)
    :type step_size: float
    :param num_leapfrog_steps: HMC number of leapfrog integration steps (L)
    :type num_leapfrog_steps: int
    :param num_burnin_steps: number of brun-in steps
    :type num_burnin_steps: int
    :param num_results: number of sampled results
    :type num_results: int
    :param seq_len: length of the sub-chains (see `sample_chain_sequentially`)
    :type seq_len: int
    :param current_state: initial state of the chain
    :type current_state: list[tf.Tensor] | None
    :param resume: previous kernel results if this chain needs to be resumed
    :type resume: Any | None
    :param sampler: type of kernel to use: "hmc" | "nuts"
    :type sampler: str
    :param log_dir: directory to use for logging
    :type log_dir: str | Path
    :param step_size_adapter: type of step-size adapter to use: "simple" | "dual_averaging" | "none"
    :type step_size_adapter: str
    :param kwargs: extra keywords arguments for tf.mcmc.sample_chain()
    :return: results of the entire chain
    :rtype: (list[tf.Tensor], list[tf.Tensor], Any, Any)
    """
    err = "Either current_state or resume is required when calling run_hmc"
    assert current_state is not None or resume is not None, err

    summary_writer = tf.summary.create_file_writer(str(log_dir))

    step_size_adapter = {
        "simple": tfp.mcmc.SimpleStepSizeAdaptation,
        "dual_averaging": tfp.mcmc.DualAveragingStepSizeAdaptation,
        "none": None
    }[step_size_adapter]
    if sampler == "nuts":
        kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size)
        if step_size_adapter is not None:
            adaptive_kernel = step_size_adapter(
                kernel,
                num_adaptation_steps=num_burnin_steps,
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                    step_size=new_step_size
                ),
                step_size_getter_fn=lambda pkr: pkr.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
            )
        else:
            adaptive_kernel = kernel
    elif sampler == "hmc":
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
        )
        if step_size_adapter is not None:
            adaptive_kernel = step_size_adapter(
                kernel, num_adaptation_steps=num_burnin_steps
            )
        else:
            adaptive_kernel = kernel

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
    chain, trace, kernel_results = sample_chain_sequentially(num_burnin_steps + num_results, seq_len, current_state, prev_kernel_results, kernel=adaptive_kernel, trace_fn=ft.partial(trace_fn, summary_freq=20), return_final_kernel_results=True, **kwargs)

    with summary_writer.as_default():
        tf.summary.trace_export(name="hmc_trace", step=step)
    summary_writer.close()

    burnin, samples = zip(*[(t[:-num_results], t[-num_results:]) for t in chain])
    return burnin, samples, trace, kernel_results


def predict_from_chain(chain, model, x_test, uncertainty="aleatoric+epistemic", n_samples=1000):
    """
    Compute the predictions with uncertainty from an HMC chain.

    :param chain: chain of model parameters as computed by run_hmc()
    :type chain: list[tf.Tensor]
    :param model: model to use for prediction
    :type model: keras.Model
    :param x_test: test inputs
    :type x_test: tuple(tf.Tensor)
    :param uncertainty: type of uncertainty to compute: "aleatoric" | "aleatoric+epistemic"
    :type uncertainty: str
    :param n_samples: number of samples from chain to use
    :type n_samples: int
    :return: list of tensor with predictions and uncertainty estimates (unstack along last axis to access each)
    :rtype: list[tf.Tensor]
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
    """
    Maximize the log-likelihood of the parameters to find the MAP estimate.

    :param target_log_prob_fn: log-likelihood function
    :type target_log_prob_fn: function
    :param state: initial random state
    :type state: list[tf.Tensor]
    :param n_iter: number of maximization steps
    :type n_iter: int
    :param save_every: callback frequency to find MAP
    :type save_every: int
    :param callbacks: function logging the log-likelihood on the desired dataset to find the optimal one
    :return: trace of parameters and log-likelihoods
    :rtype: (list[list[tf.Tensor]], list[list[float]])
    """
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
    """
    Find the optimal value of the MAP estimate.

    :param map_trace: trace of parameters computed with get_map_trace()
    :type map_trace: list[list[tf.Tensor]]
    :param map_log_probs: trace of log-likelihoods computed with get_map_trace()
    :type map_log_probs: list[list[float]]
    :return: parameter achieving highest log-likelihood
    :rtype: list[tf.Tensor]
    """
    # map_log_probs[0/1]: train/test log probability
    test_set_max_log_prob_idx = np.argmax(map_log_probs[0])
    # Return MAP params that achieved highest test set likelihood.
    return map_trace[test_set_max_log_prob_idx]


def nest_concat(*args, axis=0):
    """
    Utility function for concatenating a new Markov chain or trace with
    older ones when resuming a previous calculation.
    """
    with tf.device('/CPU:0'):
        res = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=axis), *args)
    return res
