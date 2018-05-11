import tensorflow as tf


class SimulationCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, compiler, policy):
        self._compiler = compiler
        self._policy = policy

    @property
    def state_size(self):
        return self._compiler.state_size

    @property
    def output_size(self):
        return 1

    @property
    def graph(self):
        return self._compiler.graph

    def initial_state(self, batch_size):
        return self._compiler.compile_initial_state(batch_size)

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

    def __init__(self, compiler, policy):
        self._cell = SimulationCell(compiler, policy)

    @property
    def graph(self):
        return self._cell.graph

    @property
    def input_size(self):
        return 1

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def _timesteps(self, horizon, batch_size):
        start, limit, delta = horizon-1, -1, -1
        timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
        timesteps_range = tf.expand_dims(timesteps_range, -1)
        batch_timesteps = tf.stack([timesteps_range] * batch_size)
        return batch_timesteps

    def trajectory(self, horizon, batch_size):
        initial_state = self._cell.initial_state(batch_size)

        with self.graph.as_default():
            inputs = self._timesteps(horizon, batch_size)
            outputs, final_state = tf.nn.dynamic_rnn(
                                    self._cell,
                                    inputs,
                                    initial_state=initial_state,
                                    dtype=tf.float32,
                                    scope="trajectory")
        return outputs
