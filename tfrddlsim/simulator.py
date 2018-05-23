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
    def action_size(self):
        return self._compiler.action_size

    @property
    def output_size(self):
        return (self.state_size, self.action_size, 1)

    @property
    def graph(self):
        return self._compiler.graph

    def initial_state(self):
        return self._compiler.compile_initial_state(self._batch_size)

    def __call__(self, input, state, scope=None):

        action = self._policy(state, input)

        transition_scope = self._compiler.transition_scope(state, action)
        next_state = self._compiler.compile_cpfs(transition_scope, self._batch_size)

        next_state_scope = dict(next_state)
        next_state = tuple(fluent.tensor for _, fluent in next_state)

        transition_scope.update(next_state_scope)
        reward = self._compiler.compile_reward(transition_scope)
        reward = reward.tensor

        output_next_state = self._output(next_state)
        output_action = self._output(action)
        output = (output_next_state, output_action, reward)

        return (output, next_state)

    @classmethod
    def _output(cls, tensors):
        tensor2float = lambda t: t if t.dtype == tf.float32 else tf.cast(t, tf.float32)
        return tuple(map(tensor2float, tensors))


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
            outputs, _ = tf.nn.dynamic_rnn(
                                self._cell,
                                inputs,
                                initial_state=initial_state,
                                dtype=tf.float32,
                                scope="trajectory")
            states, actions, rewards = outputs

            # fluent types
            state_dtype = self._cell._compiler.state_dtype
            states = self._output(states, state_dtype)
            action_dtype = self._cell._compiler.action_dtype
            actions = self._output(actions, action_dtype)

            outputs = (states, actions, rewards)

        return outputs

    def run(self, trajectory):
        with tf.Session(graph=self.graph) as sess:
            states, actions, rewards = sess.run(trajectory)

        # fluent names
        state_fluent_ordering = self._cell._compiler.state_fluent_ordering
        action_fluent_ordering = self._cell._compiler.action_fluent_ordering
        states = tuple(zip(state_fluent_ordering, states))
        actions = tuple(zip(action_fluent_ordering, actions))

        return states, actions, rewards

    @classmethod
    def _output(cls, tensors, dtypes):
        outputs = []
        for t, dtype in zip(tensors, dtypes):
            t = t[0]
            if t.dtype != dtype:
                t = tf.cast(t, dtype)
            outputs.append(t)
        return tuple(outputs)
