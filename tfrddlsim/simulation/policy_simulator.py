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


import rddl2tf
from rddl2tf.compiler import Compiler
from rddl2tf.fluent import TensorFluent
from tfrddlsim.policy import Policy

import numpy as np
import tensorflow as tf


from typing import Iterable, Sequence, Optional, Tuple, Union

Shape = Sequence[int]
FluentPair = Tuple[str, TensorFluent]

NonFluentsTensor = Sequence[tf.Tensor]
StateTensor = Sequence[tf.Tensor]
StatesTensor = Sequence[tf.Tensor]
ActionsTensor = Sequence[tf.Tensor]
IntermsTensor = Sequence[tf.Tensor]

NonFluentsArray = Sequence[np.array]
StateArray = Sequence[np.array]
StatesArray = Sequence[np.array]
ActionsArray = Sequence[np.array]
IntermsArray = Sequence[np.array]

CellOutput = Tuple[StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]
CellState = Sequence[tf.Tensor]

TrajectoryOutput = Tuple[StateTensor, StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]
SimulationOutput = Tuple[NonFluentsArray, StateArray, StatesArray, ActionsArray, IntermsArray, np.array]


class PolicySimulationCell(tf.nn.rnn_cell.RNNCell):
    '''SimulationCell implements a 1-step MDP transition cell.

    It extends`tf.nn.rnn_cell.RNNCell` for simulating an MDP transition for a given policy.
    The cell input is the timestep. The hidden state is the factored MDP state.
    The cell output is the tuple of MDP fluents (next-state, action, interm, rewards).

    Note:
        All fluents are represented in factored form as Tuple[tf.Tensors].

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        policy (:obj:`tfrddlsim.policy.Policy`): MDP Policy.
        batch_size (int): The size of the simulation batch.
    '''

    def __init__(self, compiler: Compiler, policy: Policy, batch_size: int) -> None:
        self._compiler = compiler
        self._policy = policy
        self._batch_size = batch_size

    @property
    def graph(self) -> tf.Graph:
        '''Returns the computation graph.'''
        return self._compiler.graph

    @property
    def state_size(self) -> Sequence[Shape]:
        '''Returns the MDP state size.'''
        return self._sizes(self._compiler.rddl.state_size)

    @property
    def action_size(self) -> Sequence[Shape]:
        '''Returns the MDP action size.'''
        return self._sizes(self._compiler.rddl.action_size)

    @property
    def interm_size(self) -> Sequence[Shape]:
        '''Returns the MDP intermediate state size.'''
        return self._sizes(self._compiler.rddl.interm_size)

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int]:
        '''Returns the simulation cell output size.'''
        return (self.state_size, self.action_size, self.interm_size, 1)

    def initial_state(self) -> StateTensor:
        '''Returns the initial state tensor.'''
        s0 = []
        for fluent in self._compiler.compile_initial_state(self._batch_size):
            s0.append(self._output_size(fluent))
        s0 = tuple(s0)
        return s0

    def __call__(self,
            input: tf.Tensor,
            state: Sequence[tf.Tensor],
            scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:
        '''Returns the simulation cell for the given `input` and `state`.

        The cell returns states, actions and interms as a
        sequence of tensors (i.e., all representations are factored).
        The reward is an 1-dimensional tensor.

        Note:
            All tensors have shape: (batch_size, fluent_shape).

        Args:
            input (tf.Tensor): The current MDP timestep.
            state (Sequence[tf.Tensor]): State fluents in canonical order.
            scope (Optional[str]): Scope for operations in graph.

        Returns:
            Tuple[CellOutput, CellState]: (output, next_state).
        '''
        # action
        action = self._policy(state, input)

        # next state
        transition_scope = self._compiler.transition_scope(state, action)
        interm_fluents, next_state_fluents = self._compiler.compile_cpfs(transition_scope, self._batch_size)
        next_state = tuple(self._tensors(next_state_fluents))

        # reward
        transition_scope.update(next_state_fluents)
        reward = self._compiler.compile_reward(transition_scope)
        reward = self._output_size(reward.tensor)

        # outputs
        output_interm_state = self._output(interm_fluents)
        output_next_state = self._output(next_state_fluents)
        output_action = tuple(self._dtype(self._output_size(tensor)) for tensor in action)
        output = (output_next_state, output_action, output_interm_state, reward)

        return (output, next_state)

    @classmethod
    def _sizes(cls, sizes: Sequence[Shape]) -> Sequence[Union[Shape, int]]:
        return tuple(sz if sz != () else (1,) for sz in sizes)

    @classmethod
    def _output_size(cls, tensor):
        if tensor.shape.ndims == 1:
            tensor = tf.expand_dims(tensor, -1)
        return tensor

    @classmethod
    def _tensors(cls, fluents: Sequence[FluentPair]) -> Iterable[tf.Tensor]:
        '''Yields the `fluents`' tensors.'''
        for _, fluent in fluents:
            tensor = cls._output_size(fluent.tensor)
            yield tensor

    @classmethod
    def _dtype(cls, tensor: tf.Tensor) -> tf.Tensor:
        '''Converts `tensor` to tf.float32 datatype if needed.'''
        if tensor.dtype != tf.float32:
            tensor = tf.cast(tensor, tf.float32)
        return tensor

    @classmethod
    def _output(cls, fluents: Sequence[FluentPair]) -> Sequence[tf.Tensor]:
        '''Returns output tensors for `fluents`.'''
        return tuple(cls._dtype(t) for t in cls._tensors(fluents))


class PolicySimulator(object):
    '''Simulator class samples MDP trajectories in the computation graph.

    It implements the n-step MDP trajectory simulator using dynamic unrolling
    in a recurrent model. Its inputs are the MDP initial state and the number
    of timesteps in the horizon.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler.
        policy (:obj:`tfrddlsim.policy.Policy`): MDP Policy.
        batch_size (int): The size of the simulation batch.
    '''

    def __init__(self, compiler: Compiler, policy: Policy, batch_size: int) -> None:
        self._cell = PolicySimulationCell(compiler, policy, batch_size)
        self._non_fluents = [fluent.tensor for _, fluent in compiler.compile_non_fluents()]

    @property
    def graph(self):
        '''Returns the computation graph.'''
        return self._cell.graph

    @property
    def batch_size(self) -> int:
        '''Returns the size of the simulation batch.'''
        return self._cell._batch_size

    @property
    def input_size(self) -> int:
        '''Returns the simulation input size (e.g., timestep).'''
        return 1

    @property
    def state_size(self) -> Sequence[Shape]:
        '''Returns the MDP state size.'''
        return self._cell.state_size

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int]:
        '''Returns the simulation output size.'''
        return self._cell.output_size

    def timesteps(self, horizon: int) -> tf.Tensor:
        '''Returns the input tensor for the given `horizon`.'''
        start, limit, delta = horizon - 1, -1, -1
        timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
        timesteps_range = tf.expand_dims(timesteps_range, -1)
        batch_timesteps = tf.stack([timesteps_range] * self.batch_size)
        return batch_timesteps

    def trajectory(self,
            horizon: int,
            initial_state: Optional[StateTensor] = None) -> TrajectoryOutput:
        '''Returns the ops for the trajectory generation with given `horizon`
        and `initial_state`.

        The simulation returns states, actions and interms as a
        sequence of tensors (i.e., all representations are factored).
        The reward is a batch sized tensor.
        The trajectoty output is a tuple: (initial_state, states, actions, interms, rewards).
        If initial state is None, use default compiler's initial state.

        Note:
            All tensors have shape: (batch_size, horizon, fluent_shape).
            Except initial state that has shape: (batch_size, fluent_shape).

        Args:
            horizon (int): The number of simulation timesteps.
            initial_state (Optional[Sequence[tf.Tensor]]): The initial state tensors.

        Returns:
            Tuple[StateTensor, StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]: Trajectory output tuple.
        '''
        if initial_state is None:
            initial_state = self._cell.initial_state()

        with self.graph.as_default():
            self.inputs = self.timesteps(horizon)
            outputs, _ = tf.nn.dynamic_rnn(
                                self._cell,
                                self.inputs,
                                initial_state=initial_state,
                                dtype=tf.float32,
                                scope="trajectory")
            states, actions, interms, rewards = outputs

            # fluent types
            state_dtype = map(rddl2tf.utils.range_type_to_dtype, self._cell._compiler.rddl.state_range_type)
            states = self._output(states, state_dtype)
            interm_dtype = map(rddl2tf.utils.range_type_to_dtype, self._cell._compiler.rddl.interm_range_type)
            interms = self._output(interms, interm_dtype)
            action_dtype = map(rddl2tf.utils.range_type_to_dtype, self._cell._compiler.rddl.action_range_type)
            actions = self._output(actions, action_dtype)

            outputs = (initial_state, states, actions, interms, rewards)

        return outputs

    def run(self,
            horizon: int,
            initial_state: Optional[StateTensor] = None) -> SimulationOutput:
        '''Builds the MDP graph and simulates in batch the trajectories
        with given `horizon`. Returns the non-fluents, states, actions, interms
        and rewards. Fluents and non-fluents are returned in factored form.

        Note:
            All output arrays have shape: (batch_size, horizon, fluent_shape).
            Except initial state that has shape: (batch_size, fluent_shape).

        Args:
            horizon (int): The number of timesteps in the simulation.
            initial_state (Optional[Sequence[tf.Tensor]]): The initial state tensors.

        Returns:
            Tuple[NonFluentsArray, StatesArray, ActionsArray, IntermsArray, np.array]: Simulation ouput tuple.
        '''
        trajectory = self.trajectory(horizon, initial_state)

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            non_fluents = sess.run(self._non_fluents)
            initial_state, states, actions, interms, rewards = sess.run(trajectory)

        # non-fluents
        non_fluent_ordering = self._cell._compiler.rddl.domain.non_fluent_ordering
        non_fluents = tuple(zip(non_fluent_ordering, non_fluents))

        # states
        state_fluent_ordering = self._cell._compiler.rddl.domain.state_fluent_ordering
        states = tuple(zip(state_fluent_ordering, states))

        # interms
        interm_fluent_ordering = self._cell._compiler.rddl.domain.interm_fluent_ordering
        interms = tuple(zip(interm_fluent_ordering, interms))

        # actions
        action_fluent_ordering = self._cell._compiler.rddl.domain.action_fluent_ordering
        actions = tuple(zip(action_fluent_ordering, actions))

        # rewards
        rewards = np.squeeze(rewards)

        outputs = (non_fluents, initial_state, states, actions, interms, rewards)

        return outputs

    @classmethod
    def _output(cls,
            tensors: Sequence[tf.Tensor],
            dtypes: Sequence[tf.DType]) -> Sequence[tf.Tensor]:
        '''Converts `tensors` to the corresponding `dtypes`.'''
        outputs = []
        for tensor, dtype in zip(tensors, dtypes):
            tensor = tensor[0]
            if tensor.dtype != dtype:
                tensor = tf.cast(tensor, dtype)
            outputs.append(tensor)
        return tuple(outputs)
