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


import tensorflow as tf


class ActionSimulationCell(tf.nn.rnn_cell.RNNCell):
    '''ActionSimulationCell implements an MDP transition cell.

    It extends a RNNCell in order to simulate the next state,
    given the current state and action. The cell input is the
    action fluents and the cell output is the next state fluents.

    Note:
        All fluents are represented in factored form as Tuple[tf.Tensors].

    Args:
        compiler (:obj:`tfrddlsim.compiler.Compiler`): RDDL2TensorFlow compiler.
    '''

    def __init__(self, compiler, batch_size=1):
        self._compiler = compiler
        self._batch_size = batch_size

    @property
    def state_size(self):
        return self._compiler.state_size

    @property
    def output_size(self):
        return (1,)

    def __call__(self, inputs, state, scope=None):
        action = inputs

        transition_scope = self._compiler.transition_scope(state, action)
        _, next_state_fluents = self._compiler.compile_cpfs(transition_scope, self._batch_size)

        next_state_scope = dict(next_state_fluents)
        transition_scope.update(next_state_scope)
        reward = self._compiler.compile_reward(transition_scope)

        next_state = self._output(next_state_fluents)
        output = (reward.tensor,)

        return (output, next_state)

    @classmethod
    def _output(cls, fluents):
        '''Converts fluents to tensors with datatype tf.float32.'''
        output = []
        for _, fluent in fluents:
            tensor = fluent.tensor
            if tensor.dtype != tf.float32:
                tensor = tf.cast(tensor, tf.float32)
            output.append(tensor)
        return tuple(output)
