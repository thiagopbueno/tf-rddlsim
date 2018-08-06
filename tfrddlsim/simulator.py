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


from tfrddlsim.compiler import Compiler
from tfrddlsim.policy import Policy

import numpy as np
import tensorflow as tf


from typing import Sequence, Optional, Tuple

Shape = Sequence[int]

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


class SimulationCell(tf.nn.rnn_cell.RNNCell):
    '''SimulationCell implements a 1-step MDP transition cell.

    It extends`tf.nn.rnn_cell.RNNCell` for simulating an MDP transition.
    The cell input is the timestep. The hidden state is the factored MDP state.
    The cell output is the tuple of MDP fluents (state, action, interm, rewards).

    Args:
        compiler (:obj:`tfrddlsim.compiler.Compiler`): RDDL2TensorFlow compiler.
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
        return self._compiler.state_size

    @property
    def action_size(self) -> Sequence[Shape]:
        '''Returns the MDP action size.'''
        return self._compiler.action_size

    @property
    def interm_size(self) -> Sequence[Shape]:
        '''Returns the MDP intermediate state size.'''
        return self._compiler.interm_size

    @property
    def output_size(self) -> Tuple[Sequence[Shape], Sequence[Shape], Sequence[Shape], int]:
        '''Returns the simulation cell output size.'''
        return (self.state_size, self.action_size, self.interm_size, 1)

    def initial_state(self) -> tf.Tensor:
        '''Returns the initial state tensor.'''
        return self._compiler.compile_initial_state(self._batch_size)

    def __call__(self,
            input: tf.Tensor,
            state: Sequence[tf.Tensor],
            scope: Optional[str] = None) -> Tuple[CellOutput, CellState]:
        '''Returns the simulation cell for the given `input` and `state`.

        The cell returns states, actions and interms as a
        sequence of tensors (i.e., all representations are factored).
        The reward is an n-dimensional tensor.

        Note:
            All tensors have shape: (batch_size, fluent_shape).

        Args:
            input (tf.Tensor): The current MDP timestep.
            state (Sequence[tf.Tensor]): State fluents in canonical order.
            scope (Optional[str]): Scope for operations in graph.

        Returns:
            Tuple[CellOutput, CellState]: (output, next_state).
        '''
        action = self._policy(state, input)

        transition_scope = self._compiler.transition_scope(state, action)
        interm_fluents, next_state_fluents = self._compiler.compile_cpfs(transition_scope, self._batch_size)

        intermediate_state = tuple(fluent.tensor for _, fluent in interm_fluents)
        next_state = tuple(fluent.tensor for _, fluent in next_state_fluents)

        next_state_scope = dict(next_state_fluents)
        transition_scope.update(next_state_scope)
        reward = self._compiler.compile_reward(transition_scope)
        reward = reward.tensor

        output_next_state = self._output(next_state)
        output_interm_state = self._output(intermediate_state)
        output_action = self._output(action)
        output = (output_next_state, output_action, output_interm_state, reward)

        return (output, next_state)

    @classmethod
    def _output(cls, tensors: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
        '''Converts tensors to datatype tf.float32.'''
        tensor2float = lambda t: t if t.dtype == tf.float32 else tf.cast(t, tf.float32)
        return tuple(map(tensor2float, tensors))


class Simulator(object):
    '''Simulator class samples MDP trajectories in the computation graph.

    It implements the n-step MDP trajectory simulator using dynamic unrolling
    in a recurrent model. Its inputs are the MDP initial state and the number
    of timesteps in the horizon.

    Args:
        compiler (:obj:`tfrddlsim.compiler.Compiler`): RDDL2TensorFlow compiler.
        policy (:obj:`tfrddlsim.policy.Policy`): MDP Policy.
        batch_size (int): The size of the simulation batch.
    '''

    def __init__(self, compiler: Compiler, policy: Policy, batch_size: int) -> None:
        self._cell = SimulationCell(compiler, policy, batch_size)
        self._non_fluents = [fluent.tensor for _, fluent in compiler.non_fluents]

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
        with self.graph.as_default():
            start, limit, delta = horizon - 1, -1, -1
            timesteps_range = tf.range(start, limit, delta, dtype=tf.float32)
            timesteps_range = tf.expand_dims(timesteps_range, -1)
            batch_timesteps = tf.stack([timesteps_range] * self.batch_size)
            return batch_timesteps

    def trajectory(self, horizon: int) -> TrajectoryOutput:
        '''Returns the ops for the trajectory generation with given `horizon`.

        The simulation returns states, actions and interms as a
        sequence of tensors (i.e., all representations are factored).
        The reward is an n-dimensional tensor.
        The trajectoty output is a tuple: (initial_state, states, actions, interms, rewards).

        Note:
            All tensors have shape: (batch_size, horizon, fluent_shape).
            Except initial state that has shape: (batch_size, fluent_shape).

        Args:
            horizon (int): The number of simulation timesteps.

        Returns:
            Tuple[StateTensor, StatesTensor, ActionsTensor, IntermsTensor, tf.Tensor]: Trajectory output tuple.
        '''
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

            outputs = (initial_state, states, actions, interms, rewards)

        return outputs

    def run(self, horizon: int = 40) -> SimulationOutput:
        '''Builds the MDP graph and simulates in batch the trajectories
        with given `horizon`. Returns the non-fluents, states, actions, interms
        and rewards. Fluents and non-fluents are returned in factored form.

        Note:
            All output arrays have shape: (batch_size, horizon, fluent_shape).
            Except initial state that has shape: (batch_size, fluent_shape).

        Args:
            horizon (int): The number of timesteps in the simulation.

        Returns:
            Tuple[NonFluentsArray, StatesArray, ActionsArray, IntermsArray, np.array]: Simulation ouput tuple.
        '''
        trajectory = self.trajectory(horizon)

        with tf.Session(graph=self.graph) as sess:
            non_fluents = sess.run(self._non_fluents)
            initial_state, states, actions, interms, rewards = sess.run(trajectory)

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

        outputs = (non_fluents, initial_state, states, actions, interms, rewards)

        return outputs

    @classmethod
    def _output(cls,
            tensors: Sequence[tf.Tensor],
            dtypes: Sequence[tf.DType]) -> Sequence[tf.Tensor]:
        '''Converts `tensors` to the corresponding `dtypes`.'''
        outputs = []
        for t, dtype in zip(tensors, dtypes):
            t = t[0]
            if t.dtype != dtype:
                t = tf.cast(t, dtype)
            outputs.append(t)
        return tuple(outputs)
