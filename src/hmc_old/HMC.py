import tensorflow as tf
from tensorflow import keras
from typing import NamedTuple


class HMCState(NamedTuple):
    """
    Basic NamedTuple extension to represent a given state of the HMC walk.
    """
    position: tf.Variable
    log_gamma: tf.Variable
    log_lambda: tf.Variable
    momentum: tf.Tensor
    log_gamma_p: tf.Tensor
    log_lambda_p: tf.Tensor


class HMC(keras.Model):
    def __init__(self, model: keras.Model,
                 L: int,
                 epsilon_min: float,
                 epsilon_max: float,
                 batch_size: int,
                 n_obs: int):
        """
        "Super-Model" class to train a model through HMC random walk.

        :param model: Model to train.
        :type model: keras.Model
        :param L: number of leapfrog steps.
        :type L: int
        :param epsilon: leapfrog step size.
        :type epsilon: float
        :param batch_size: size of the input batch.
        :type batch_size: int
        """
        super().__init__()
        self.model = model
        self.L = L
        self.epsilon_min = tf.constant(epsilon_min, dtype=tf.float32)
        self.epsilon_max = tf.constant(epsilon_max, dtype=tf.float32)
        self.batch_size = tf.constant(batch_size)
        self.n_observations = tf.constant(n_obs)
        self.param_num = tf.constant(sum([tf.size(var) for var in model.trainable_variables]))
        self.param_shapes = [var.shape for var in model.trainable_variables]
        self.log_gamma = tf.Variable(tf.random.normal((1,)) + 4, shape=(1,))
        self.log_lambda = tf.Variable(tf.random.normal((1,)), shape=(1,))
        self.loss = model.loss

    def init_parameters(self, inputs):
        loss = self.get_loss(inputs)
        self.log_gamma[0].assign(tf.math.log(tf.cast(self.n_observations, dtype=tf.float32) / loss))
        if self.log_gamma > 6.0:
            self.log_gamma[0].assign(6.0)
            # self.epsilon_min = tf.constant(5e-4, dtype=tf.float32)
            # self.epsilon_max = tf.constant(5e-4, dtype=tf.float32)
        parameters = self.get_model_params()
        self.log_lambda[0].assign(tf.math.log(1/tf.sqrt(tf.math.reduce_variance(parameters)/2)))
        # self.log_lambda[0].assign(-15)

    def epsilon(self, step, n_iter):
        # return self.epsilon_max * tf.exp(- tf.cast(step, tf.float32) / tf.cast(n_iter, tf.float32)
        #                                  * tf.math.log(self.epsilon_max / self.epsilon_min))
        rnd = tf.random.uniform((), tf.math.log(self.epsilon_min), tf.math.log(self.epsilon_max))
        return tf.exp(rnd)

    @staticmethod
    def probability(hamiltonian_init, hamiltonian_final):
        """
        Return the probability to accept a step.

        :param hamiltonian_init: initial Hamiltonian value
        :type hamiltonian_init: tf.Tensor
        :param hamiltonian_final: final Hamiltonian value
        :type hamiltonian_final: tf.Tensor
        :return: probability to accept the new state
        :rtype: tf.Tensor
        """
        p = tf.exp(hamiltonian_init - hamiltonian_final)
        p = tf.minimum(p, 1.0)
        return p

    def get_model_params(self):
        """
        Query the model parameters as a unique one-dimensional tensor.

        :return: current parameters.
        :rtype: tf.Tensor
        """
        params = self.model.trainable_variables
        params = [tf.reshape(param, [-1]) for param in params]
        return tf.concat(params, 0)

    def set_model_params(self, params):
        """
        Set the parameters of the model from a unique one-dimensional tensor.

        :param params: parameter values, as a one-dimensional tensor.
        :type params: tf.Tensor
        :return: None
        :rtype: None
        """
        shaped_params = []
        idx = 0
        for shape in self.param_shapes:
            size = tf.reduce_prod(shape)
            param = params[idx:(idx + size)]
            param = tf.reshape(param, shape)
            shaped_params.append(param)
            idx += size
        for var, param in zip(self.model.trainable_variables, shaped_params):
            var.assign(param)

    def get_loss(self, inputs):
        """
        Compute sub-model loss.

        :param inputs: training data, tuple (inputs, targets).
        :type inputs: (Any, Any)
        :return: loss value
        :rtype: tf.Tensor
        """
        batch, targets = inputs
        pred = self.model(batch)
        loss = self.loss(pred, targets)
        return loss

    def get_loss_and_grad(self, inputs, state=None):
        """
        Compute sub-model loss and gradient.

        :param inputs: training data, tuple (inputs, targets).
        :type inputs: (tf.Tensor, tf.Tensor)
        :param state: current state of the model.
        :type state: HMCState
        :return: loss and gradient of the given model (loss, grad); the gradient is reshaped as a one-dimensional Tensor.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        batch, targets = inputs
        if state is not None:
            self.set_model_params(state.position)
        with tf.GradientTape() as tape:
            tape.watch(batch)
            pred = self.model(batch)
            loss = self.loss(pred, targets)
        grad = tape.gradient(loss, self.model.trainable_variables)
        grad = tf.concat([tf.reshape(param, [-1]) for param in grad], 0)
        return loss, grad

    def get_hyper_grad(self, loss, state: HMCState):
        """
        Compute the gradient of λ and γ.

        :param loss: the loss value in the current state.
        :type loss: tf.Tensor
        :param state: current state of the walk.
        :type state: HMCState
        :return: gradient of γ, gradient of λ.
        :rtype: (tf.Tensor, tf.Tensor)
        """
        dlog_gamma = tf.exp(state.log_gamma) * (loss / 2.0 + 1.0) \
                     - (tf.cast(self.n_observations, tf.float32) / 2.0 + 1.0)
        dlog_lambda = tf.exp(state.log_lambda) * (tf.reduce_sum(tf.abs(state.position)) + 1.0) \
                      - (tf.cast(self.param_num, tf.float32) + 1.0)
        return dlog_gamma, dlog_lambda

    @staticmethod
    def get_hmc_grad(grad, state: HMCState):
        """
        Update the parameters' gradient for HMC setup.

        :param grad: gradient of the parameters as returned by TF.
        :type grad: tf.Tensor
        :param state: current state of the walk.
        :type state: HMCState
        :return: updated gradient.
        :rtype: tf.Tensor
        """
        w = state.position
        dw = tf.exp(state.log_gamma) / 2.0 * grad + tf.exp(state.log_lambda) * tf.sign(w)
        return dw

    def get_config(self):
        # TODO
        pass

    def state(self, momentum: tf.Tensor, log_gamma_p: tf.Tensor, log_lambda_p: tf.Tensor):
        """
        Create an instance of HMCState from current model.

        :param momentum: momentum value.
        :type momentum: tf.Tensor
        :param log_gamma_p: log(γ) value for the momentum.
        :type log_gamma_p: tf.Tensor
        :param log_lambda_p: log(λ) value for the momentum.
        :type log_lambda_p: tf.Tensor
        :return: the current state of the model
        :rtype: HMCState
        """
        return HMCState(self.get_model_params(),
                        self.log_gamma,
                        self.log_lambda,
                        momentum,
                        log_gamma_p,
                        log_lambda_p)

    def update_state(self, state: HMCState):
        """
        Update the model with a new state.

        :param state: new state of the model.
        :type state: HMCState
        :return: None
        :rtype: None
        """
        self.set_model_params(state.position)
        self.log_lambda.assign(state.log_lambda)
        self.log_gamma.assign(state.log_gamma)

    @staticmethod
    def kinetic_energy(state: HMCState):
        """
        Return the kinetic energy of a given state.

        :param state: current state of the walk.
        :type state: HMCState
        :return: kinetic energy
        :rtype: tf.Tensor
        """
        k = (tf.reduce_sum(state.momentum ** 2) + state.log_gamma_p ** 2 + state.log_lambda_p ** 2) / 2.0
        return k

    def potential_energy(self, loss, state: HMCState):
        """
        Compute the potential energy of the system for a given state.

        The potential energy is defined as the negatie log likelihood of the current state.

        :param loss: the loss value in the current state.
        :type loss: tf.Tensor
        :param state: current state of the walk.
        :type state: HMCState
        :return: potential energy.
        :rtype: tf.Tensor
        """
        w = state.position
        u = tf.exp(state.log_gamma) * (loss / 2.0 + 1.0) \
            + tf.exp(state.log_lambda) * (tf.reduce_sum(tf.abs(w)) + 1.0) \
            - (tf.cast(self.n_observations, tf.float32) / 2.0 + 1.0) * state.log_gamma \
            - (tf.cast(self.param_num, tf.float32) + 1.0) * state.log_lambda
        return u

    def hamiltonian(self, loss, state: HMCState):
        """
        Compute the Hamiltonian of the walk for a given state.

        :param loss: the loss value in the current state.
        :type loss: tf.Tensor
        :param state: current state of the walk.
        :type state: HMCState
        :return: current hamiltonian value, loss value
        :rtype: tf.Tensor
        """
        k = HMC.kinetic_energy(state)
        u = self.potential_energy(loss, state)
        return k + u, -u

    def leap_frog(self, inputs, state: HMCState, epsilon):
        """
        Run `L` steps of the leapfrog procedure to update the Hamiltonian.

        :param inputs: training data, tuple (inputs, targets).
        :type inputs: (Any, Any)
        :param state: initial state of the walk.
        :type state: HMCState
        :param epsilon: epsilon value for the leapfrog process.
        :type epsilon: tf.Tensor
        :return: final state of the walk.
        :rtype: HMCState
        """

        current_state = state

        # Half-step for the momentum
        loss, grad = self.get_loss_and_grad(inputs, current_state)
        grad = HMC.get_hmc_grad(grad, current_state)
        dlog_gamma, dlog_lambda = self.get_hyper_grad(loss, current_state)

        log_gamma_p_new = current_state.log_gamma_p - epsilon / 2.0 * dlog_gamma
        log_lambda_p_new = current_state.log_lambda_p - epsilon / 2.0 * dlog_lambda
        momentum_new = current_state.momentum - epsilon / 2.0 * grad

        for i in range(self.L):

            # Full step for the position
            log_gamma_new = current_state.log_gamma + epsilon * log_gamma_p_new
            log_lambda_new = current_state.log_lambda + epsilon * log_lambda_p_new
            position_new = current_state.position + epsilon * momentum_new

            half_state = HMCState(position_new,
                                  log_gamma_new,
                                  log_lambda_new,
                                  momentum_new,
                                  log_gamma_p_new,
                                  log_lambda_p_new)

            # Full step for the momentum, except at the end of the trajectory
            if i == self.L - 1:
                continue
            loss, grad = self.get_loss_and_grad(inputs, half_state)
            grad = HMC.get_hmc_grad(grad, half_state)
            dlog_gamma, dlog_lambda = self.get_hyper_grad(loss, half_state)

            log_gamma_p_new = half_state.log_gamma_p - epsilon * dlog_gamma
            log_lambda_p_new = half_state.log_lambda_p - epsilon * dlog_lambda
            momentum_new = half_state.momentum - epsilon * grad

            current_state = HMCState(half_state.position,
                                     half_state.log_gamma,
                                     half_state.log_lambda,
                                     momentum_new,
                                     log_gamma_p_new,
                                     log_lambda_p_new)

        # Half-step for the momentum
        loss, grad = self.get_loss_and_grad(inputs, current_state)
        grad = HMC.get_hmc_grad(grad, current_state)
        dlog_gamma, dlog_lambda = self.get_hyper_grad(loss, current_state)

        log_gamma_p_new = current_state.log_gamma_p - epsilon / 2.0 * dlog_gamma
        log_lambda_p_new = current_state.log_lambda_p - epsilon / 2.0 * dlog_lambda
        momentum_new = current_state.momentum - epsilon / 2.0 * grad


        return HMCState(
            current_state.position,
            current_state.log_gamma,
            current_state.log_lambda,
            -momentum_new,
            -log_gamma_p_new,
            -log_lambda_p_new
        )

    @tf.function
    def call(self, inputs, step, n_iter):
        """
        Run one step of HMC walk with the given model.

        :param inputs: training data, tuple (inputs, targets).
        :type inputs: (Any, Any)
        :param step: current step of the walk.
        :type step: tf.Tensor
        :param n_iter: total length of the HMC walk.
        :type n_iter: tf.Tensor
        :return: Final state of the model and logging data (Final State, Loss value, probability of new state, is new state accepted, Hamiltonian value).
        :rtype: (HMCState, tf.Tensor, tf.Tensor, bool, tf.Tensor)
        """
        epsilon = self.epsilon(step, n_iter)

        momentum = tf.random.normal((self.param_num,))
        log_gamma_p = tf.random.normal((1,))
        log_lambda_p = tf.random.normal((1,))
        init_state = self.state(momentum, log_gamma_p, log_lambda_p)

        self.set_model_params(init_state.position)
        loss_initial = self.get_loss(inputs)
        hamiltonian_initial, log_likelihood_initial = self.hamiltonian(loss_initial, init_state)

        new_state = self.leap_frog(inputs, init_state, epsilon)

        self.set_model_params(new_state.position)
        loss_final = self.get_loss(inputs)
        hamiltonian_final, log_likelihood_final = self.hamiltonian(loss_final, new_state)

        p_new_state = self.probability(hamiltonian_initial, hamiltonian_final)
        p = tf.random.uniform((1,), 0, 1)

        if p < p_new_state:
            self.update_state(new_state)
            return new_state, log_likelihood_final, p_new_state, True
        else:
            self.update_state(init_state)
            return init_state, log_likelihood_initial, p_new_state, False
