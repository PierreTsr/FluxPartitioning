import tensorflow as tf
from tensorflow import keras
from typing import NamedTuple, Any

tf.random.set_seed(1234)


class HMCState(NamedTuple):
    position: Any
    log_gamma: tf.Variable
    log_lambda: tf.Variable
    momentum: tf.Tensor
    log_gamma_p: tf.Tensor
    log_lambda_p: tf.Tensor


class HMC(keras.Model):
    def __init__(self, model: keras.Layer,
                 L: int,
                 epsilon: float,
                 batch_size: int):
        super(HMC, self).__init__()
        self.model = model
        self.L = L
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.param_num = model.trainable_variables.size
        self.log_gamma = tf.Variable(tf.random.normal((1,)))
        self.log_lambda = tf.Variable(tf.random.normal((1,)))
        self.loss = keras.losses.MeanSquaredError()

    @staticmethod
    def probability(hamiltonian_init, hamiltonian_final):
        p = tf.exp(hamiltonian_init - hamiltonian_final)
        p = tf.minimum(p, 1)
        return p

    @staticmethod
    def kinetic_energy(state: HMCState):
        k = (tf.reduce_sum(state.momentum ** 2) + state.log_gamma_p ** 2 + state.log_lambda_p ** 2) / 2
        return k

    @staticmethod
    def get_hmc_grad(grad, state: HMCState):
        w = state.position
        dw = tf.exp(state.log_gamma) / 2 * grad + tf.exp(state.log_lambda) * tf.sign(w)
        return dw

    def get_hyper_grad(self, inputs, state: HMCState):
        self.model.trainable_variables.assign(state.position)
        loss = self.get_loss(inputs)
        dlog_gamma = tf.exp(state.log_gamma) * (loss / 2 + 1) - (self.batch_size / 2 + 1)
        dlog_lambda = tf.exp(state.log_lambda) * (tf.reduce_sum(tf.abs(state.position)) + 1) - (self.param_num + 1)
        return dlog_gamma, dlog_lambda

    def call(self, inputs):
        momentum = tf.random.normal((self.param_num,))
        log_gamma_p = tf.random.normal((1,))
        log_lambda_p = tf.random.normal((1,))
        init_state = self.state(momentum, log_gamma_p, log_lambda_p)

        hamiltonian_init, loss_initial = self.hamiltonian(inputs, init_state)
        new_state = self.leap_frog(inputs, init_state)
        hamiltonian_final, loss_final = self.hamiltonian(inputs, new_state)

        p_new_state = self.probability(hamiltonian_init, hamiltonian_final)
        p = tf.random.uniform((1,), 0, 1)

        if p < p_new_state:
            final_state = new_state
            self.update_state(final_state)
            return final_state, loss_final
        else:
            final_state = init_state
            self.update_state(init_state)
            return final_state, loss_initial

    def get_config(self):
        # TODO
        pass

    def state(self, momentum: tf.Tensor, log_gamma_p: tf.Tensor, log_lambda_p: tf.Tensor):
        return HMCState(self.model.trainable_variables.copy(),
                        self.log_gamma,
                        self.log_lambda,
                        momentum,
                        log_gamma_p,
                        log_lambda_p)

    def update_state(self, state: HMCState):
        self.model.trainable_variables.assign(state.position)
        self.log_lambda = state.log_lambda
        self.log_gamma = state.log_gamma

    def get_loss(self, inputs):
        batch, targets = inputs
        pred = self.model(batch)
        loss = self.loss(pred, targets)
        return loss

    def get_loss_and_grad(self, inputs, state=None):
        batch, targets = inputs
        if state is not None:
            self.model.trainable_variables.assign(state.position)
        with tf.GradientTape() as tape:
            tape.watch(batch)
            pred = self.model(batch)
            loss = self.loss(pred, targets)
        grad = tape.gradient(loss, self.model.trainable_variables)
        return loss, grad

    def potential_energy(self, inputs, state: HMCState):
        w = state.position
        loss = self.get_loss(inputs)
        u = tf.exp(state.log_gamma) * (loss / 2 + 1) \
            + tf.exp(state.log_lambda) * (tf.reduce_sum(tf.abs(w)) + 1) \
            - (self.batch_size / 2 + 1) * state.log_gamma \
            - (self.param_num + 1) * state.log_lambda
        return u, loss

    def hamiltonian(self, inputs, state: HMCState):
        k = HMC.kinetic_energy(state)
        u, loss = self.potential_energy(inputs, state)
        return k + u, loss

    def leap_frog(self, inputs, state: HMCState):

        self.model.trainable_variables.assign(state.position)
        current_state = state
        for i in range(self.L):
            loss, grad = self.get_loss_and_grad(inputs, current_state)
            grad = HMC.get_hmc_grad(grad, current_state)
            dlog_gamma, dlog_lambda = self.get_hyper_grad(inputs, current_state)

            log_gamma_p_new = current_state.log_gamma_p - self.epsilon / 2 * dlog_gamma
            log_lambda_p_new = current_state.log_lambda_p - self.epsilon / 2 * dlog_lambda
            momentum_new = current_state.momentum - self.epsilon / 2 * grad

            log_gamma_new = current_state.log_gamma + self.epsilon * log_gamma_p_new
            log_lambda_new = current_state.log_lambda + self.epsilon * log_lambda_p_new
            position_new = current_state.position + self.epsilon * momentum_new

            half_state = HMCState(position_new,
                                  log_gamma_new,
                                  log_lambda_new,
                                  momentum_new,
                                  log_gamma_p_new,
                                  log_lambda_p_new)

            loss, grad = self.get_loss_and_grad(inputs, half_state)
            grad = HMC.get_hmc_grad(grad, half_state)
            dlog_gamma, dlog_lambda = self.get_hyper_grad(inputs, half_state)

            log_gamma_p_new = half_state.log_gamma_p - self.epsilon / 2 * dlog_gamma
            log_lambda_p_new = half_state.log_lambda_p - self.epsilon / 2 * dlog_lambda
            momentum_new = half_state.momentum - self.epsilon / 2 * grad

            current_state = HMCState(half_state.position,
                                     half_state.log_gamma,
                                     half_state.log_lambda,
                                     momentum_new,
                                     log_gamma_p_new,
                                     log_lambda_p_new)
        return current_state





