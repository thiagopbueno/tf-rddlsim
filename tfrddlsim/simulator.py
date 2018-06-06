# This file is part of tf-rddlsim.

# tf-rddlsim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-rddlsim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-rddlsim. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
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
    def interm_size(self):
        return self._compiler.interm_size

    @property
    def output_size(self):
        return (self.state_size, self.action_size, self.interm_size, 1)

    @property
    def graph(self):
        return self._compiler.graph

    def initial_state(self):
        return self._compiler.compile_initial_state(self._batch_size)

    def __call__(self, input, state, scope=None):
        action = self._policy(state, input)

        scope = self._compiler.transition_scope(state, action)
        interm_fluents, next_state_fluents = self._compiler.compile_cpfs(scope, self._batch_size)

        intermediate_state = tuple(fluent.tensor for _, fluent in interm_fluents)
        next_state = tuple(fluent.tensor for _, fluent in next_state_fluents)

        next_state_scope = dict(next_state_fluents)
        scope.update(next_state_scope)
        reward = self._compiler.compile_reward(scope)
        reward = reward.tensor

        output_next_state = self._output(next_state)
        output_interm_state = self._output(intermediate_state)
        output_action = self._output(action)
        output = (output_next_state, output_action, output_interm_state, reward)

        return (output, next_state)

    @classmethod
    def _output(cls, tensors):
        tensor2float = lambda t: t if t.dtype == tf.float32 else tf.cast(t, tf.float32)
        return tuple(map(tensor2float, tensors))


class Simulator(object):

    def __init__(self, compiler, policy, batch_size):
        self._cell = SimulationCell(compiler, policy, batch_size)
        self._non_fluents = [fluent.tensor for _, fluent in compiler.non_fluents]

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
        with self.graph.as_default():
            start, limit, delta = horizon - 1, -1, -1
            timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
            timesteps_range = tf.expand_dims(timesteps_range, -1)
            batch_timesteps = tf.stack([timesteps_range] * self.batch_size)
            return batch_timesteps

    def trajectory(self, horizon):
        initial_state = self._cell.initial_state()
        inputs = self.timesteps(horizon)

        with self.graph.as_default():
            outputs, _ = tf.nn.dynamic_rnn(
                                self._cell,
                                inputs,
                                initial_state=initial_state,
                                dtype=tf.float32,
                                scope="trajectory")
            states, actions, interms, rewards = outputs

            # fluent types
            state_dtype = self._cell._compiler.state_dtype
            states = self._output(states, state_dtype)
            interm_dtype = self._cell._compiler.interm_dtype
            interms = self._output(interms, interm_dtype)
            action_dtype = self._cell._compiler.action_dtype
            actions = self._output(actions, action_dtype)

            outputs = (states, actions, interms, rewards)

        return outputs

    def run(self, horizon=40):
        trajectory = self.trajectory(horizon)

        with tf.Session(graph=self.graph) as sess:
            non_fluents = sess.run(self._non_fluents)
            states, actions, interms, rewards = sess.run(trajectory)

        # non-fluents
        non_fluent_ordering = self._cell._compiler.non_fluent_ordering
        non_fluents = tuple(zip(non_fluent_ordering, non_fluents))

        # states
        state_fluent_ordering = self._cell._compiler.state_fluent_ordering
        states = tuple(zip(state_fluent_ordering, states))

        # interms
        interm_fluent_ordering = self._cell._compiler.interm_fluent_ordering
        interms = tuple(zip(interm_fluent_ordering, interms))

        # actions
        action_fluent_ordering = self._cell._compiler.action_fluent_ordering
        actions = tuple(zip(action_fluent_ordering, actions))

        # rewards
        rewards = np.squeeze(rewards)

        return non_fluents, states, actions, interms, rewards

    @classmethod
    def _output(cls, tensors, dtypes):
        outputs = []
        for t, dtype in zip(tensors, dtypes):
            t = t[0]
            if t.dtype != dtype:
                t = tf.cast(t, dtype)
            outputs.append(t)
        return tuple(outputs)
