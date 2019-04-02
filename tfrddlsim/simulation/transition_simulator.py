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


from rddl2tf.compiler import Compiler
from rddl2tf.fluent import TensorFluent

import tensorflow as tf

from typing import Optional, Sequence, Tuple

Shape = Sequence[int]
FluentPair = Tuple[str, TensorFluent]

ActionTensor = Sequence[tf.Tensor]
StateTensor = Sequence[tf.Tensor]
StatesTensor = Sequence[tf.Tensor]
ActionsTensor = Sequence[tf.Tensor]
IntermsTensor = Sequence[tf.Tensor]


CellOutput = Tuple[StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]
CellState = Sequence[tf.Tensor]


class ActionSimulationCell(tf.nn.rnn_cell.RNNCell):
    '''ActionSimulationCell implements an MDP transition cell.

    It extends a RNNCell in order to simulate the next state,
    given the current state and action. The cell input is the
    action fluents and the cell output is the next state fluents.

    Note:
        All fluents are represented in factored form as Sequence[tf.Tensors].

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        batch_size (int): The simulation batch size.
    '''

    def __init__(self, compiler: Compiler, batch_size: int = 1) -> None:
        self._compiler = compiler
        self._batch_size = batch_size

    @property
    def state_size(self) -> Sequence[Shape]:
        '''Returns the MDP state size.'''
        return self._compiler.rddl.state_size

    @property
    def action_size(self) -> Sequence[Shape]:
        '''Returns the MDP action size.'''
        return self._compiler.rddl.action_size

    @property
    def interm_size(self) -> Sequence[Shape]:
        '''Returns the MDP intermediate state size.'''
        return self._compiler.rddl.interm_size

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int]:
        '''Returns the simulation cell output size.'''
        return (self.state_size, self.action_size, self.interm_size, 1)

    def __call__(self,
            inputs: ActionTensor,
            state: StateTensor,
            scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:
        '''Returns the transition simulation cell for the given `input` and `state`.

        The cell outputs the reward as an 1-dimensional tensor, and
        the next state as a tuple of tensors.

        Note:
            All tensors have shape: (batch_size, fluent_shape).

        Args:
            input (tf.Tensor): The current action.
            state (Sequence[tf.Tensor]): The current state.
            scope (Optional[str]): Operations' scope in computation graph.

        Returns:
            Tuple[CellOutput, CellState]: (output, next_state).
        '''
        # action
        action = inputs

        # next state
        transition_scope = self._compiler.transition_scope(state, action)
        interm_fluents, next_state_fluents = self._compiler.compile_cpfs(transition_scope, self._batch_size)

        # reward
        next_state_scope = dict(next_state_fluents)
        transition_scope.update(next_state_scope)
        reward = self._compiler.compile_reward(transition_scope)
        reward = self._output_size(reward.tensor)

        # outputs
        interm_state = self._output(interm_fluents)
        next_state = self._output(next_state_fluents)
        output = (next_state, action, interm_state, reward)

        return (output, next_state)

    @classmethod
    def _output_size(cls, tensor):
        if tensor.shape.ndims == 1:
            tensor = tf.expand_dims(tensor, -1)
        return tensor

    @classmethod
    def _output(cls, fluents: Sequence[FluentPair]) -> Sequence[tf.Tensor]:
        '''Converts `fluents` to tensors with datatype tf.float32.'''
        output = []
        for _, fluent in fluents:
            tensor = fluent.tensor
            if tensor.dtype != tf.float32:
                tensor = tf.cast(tensor, tf.float32)
            output.append(tensor)
        return tuple(output)
