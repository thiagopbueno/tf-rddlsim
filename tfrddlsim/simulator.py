import tensorflow as tf


class SimulationCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, compiler, policy, batch_size):
        self._compiler = compiler
        self._policy = policy
        self._batch_size = batch_size

    @property
    def state_size(self):
        return self._compiler.state_size

    @property
    def output_size(self):
        return 1

    @property
    def graph(self):
        return self._compiler.graph

    def initial_state(self):
        return self._compiler.compile_initial_state(self._batch_size)

    def __call__(self, input, state, scope=None):

        action = self._policy(state, input)

        non_fluent_scope = self._compiler.non_fluents_scope()
        state_scope = self._compiler.state_scope(state)
        action_scope = self._compiler.action_scope(action)

        transition_scope = {}
        transition_scope.update(non_fluent_scope)
        transition_scope.update(state_scope)
        transition_scope.update(action_scope)
        next_state = self._compiler.compile_cpfs(transition_scope)

        next_state_scope = dict(next_state)
        next_state = tuple(fluent.tensor for _, fluent in next_state)

        transition_scope.update(next_state_scope)
        reward = self._compiler.compile_reward(transition_scope)
        output = tf.expand_dims(reward.tensor, [-1])

        return (output, next_state)


class Simulator(object):

    def __init__(self, compiler, policy, batch_size):
        self._cell = SimulationCell(compiler, policy, batch_size)

    @property
    def graph(self):
        return self._cell.graph

    @property
    def batch_size(self):
        return self._cell._batch_size

    @property
    def input_size(self):
        return 1

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def timesteps(self, horizon):
        start, limit, delta = horizon-1, -1, -1
        timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
        timesteps_range = tf.expand_dims(timesteps_range, -1)
        batch_timesteps = tf.stack([timesteps_range] * self.batch_size)
        return batch_timesteps

    def trajectory(self, horizon):
        initial_state = self._cell.initial_state()

        with self.graph.as_default():
            inputs = self.timesteps(horizon)
            outputs, final_state = tf.nn.dynamic_rnn(
                                    self._cell,
                                    inputs,
                                    initial_state=initial_state,
                                    dtype=tf.float32,
                                    scope="trajectory")
        return outputs
