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
from tfrddlsim.viz.abstract_visualizer import Visualizer

import numpy as np

from typing import List, Sequence, Optional, Tuple, Union
Value = Union[bool, int, float, np.array]
NonFluents = Sequence[Tuple[str, Value]]
Fluents = Sequence[Tuple[str, np.array]]


class GenericVisualizer(Visualizer):
    '''GenericVisualizer is a generic text-based trajectory visualizer.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler
        verbose (bool): Verbosity flag
    '''

    def __init__(self, compiler: Compiler, verbose: bool) -> None:
        super().__init__(compiler, verbose)

    def render(self,
            trajectories: Tuple[NonFluents, Fluents, Fluents, Fluents, np.array],
            batch: Optional[int] = None) -> None:
        '''Prints the simulated `trajectories`.

        Args:
            trajectories: NonFluents, states, actions, interms and rewards.
            batch: Number of batches to render.
        '''
        self._render_trajectories(trajectories)

    def _render_trajectories(self,
            trajectories: Tuple[NonFluents, Fluents, Fluents, Fluents, np.array]) -> None:
        '''Prints the first batch of simulated `trajectories`.

        Args:
            trajectories: NonFluents, states, actions, interms and rewards.
        '''
        if self._verbose:
            non_fluents, initial_state, states, actions, interms, rewards = trajectories
            shape = states[0][1].shape
            batch_size, horizon, = shape[0], shape[1]
            states = [(s[0], s[1][0]) for s in states]
            interms = [(f[0], f[1][0]) for f in interms]
            actions = [(a[0], a[1][0]) for a in actions]
            rewards = np.reshape(rewards, [batch_size, horizon])[0]
            self._render_batch(non_fluents, states, actions, interms, rewards)

    def _render_batch(self,
            non_fluents: NonFluents,
            states: Fluents, actions: Fluents, interms: Fluents,
            rewards: np.array,
            horizon: Optional[int] = None) -> None:
        '''Prints `non_fluents`, `states`, `actions`, `interms` and `rewards`
        for given `horizon`.

        Args:
            states (Sequence[Tuple[str, np.array]]): A state trajectory.
            actions (Sequence[Tuple[str, np.array]]): An action trajectory.
            interms (Sequence[Tuple[str, np.array]]): An interm state trajectory.
            rewards (np.array): Sequence of rewards (1-dimensional array).
            horizon (Optional[int]): Number of timesteps.
        '''
        if horizon is None:
            horizon = len(states[0][1])
            self._render_round_init(horizon, non_fluents)
            for t in range(horizon):
                s = [(s[0], s[1][t]) for s in states]
                f = [(f[0], f[1][t]) for f in interms]
                a = [(a[0], a[1][t]) for a in actions]
                r = rewards[t]
                self._render_timestep(t, s, a, f, r)
            self._render_round_end(rewards)

    def _render_timestep(self,
            t: int,
            s: Fluents, a: Fluents, f: Fluents,
            r: np.float32) -> None:
        '''Prints fluents and rewards for the given timestep `t`.

        Args:
            t (int): timestep
            s (Sequence[Tuple[str], np.array]: State fluents.
            a (Sequence[Tuple[str], np.array]: Action fluents.
            f (Sequence[Tuple[str], np.array]: Interm state fluents.
            r (np.float32): Reward.
        '''
        print("============================")
        print("TIME = {}".format(t))
        print("============================")
        fluent_variables = self._compiler.rddl.action_fluent_variables
        self._render_fluent_timestep('action', a, fluent_variables)
        fluent_variables = self._compiler.rddl.interm_fluent_variables
        self._render_fluent_timestep('interms', f, fluent_variables)
        fluent_variables = self._compiler.rddl.state_fluent_variables
        self._render_fluent_timestep('states', s, fluent_variables)
        self._render_reward(r)

    def _render_fluent_timestep(self,
            fluent_type: str,
            fluents: Sequence[Tuple[str, np.array]],
            fluent_variables: Sequence[Tuple[str, List[str]]]) -> None:
        '''Prints `fluents` of given `fluent_type` as list of instantiated variables
        with corresponding values.

        Args:
            fluent_type (str): Fluent type.
            fluents (Sequence[Tuple[str, np.array]]): List of pairs (fluent_name, fluent_values).
            fluent_variables (Sequence[Tuple[str, List[str]]]): List of pairs (fluent_name, args).
        '''
        for fluent_pair, variable_list in zip(fluents, fluent_variables):
            name, fluent = fluent_pair
            _, variables = variable_list
            print(name)
            fluent = fluent.flatten()
            for variable, value in zip(variables, fluent):
                print('- {}: {} = {}'.format(fluent_type, variable, value))
        print()

    def _render_reward(self, r: np.float32) -> None:
        '''Prints reward `r`.'''
        print("reward = {:.4f}".format(float(r)))
        print()

    def _render_round_init(self, horizon: int, non_fluents: NonFluents) -> None:
        '''Prints round init information about `horizon` and `non_fluents`.'''
        print('*********************************************************')
        print('>>> ROUND INIT, horizon = {}'.format(horizon))
        print('*********************************************************')
        fluent_variables = self._compiler.rddl.non_fluent_variables
        self._render_fluent_timestep('non-fluents', non_fluents, fluent_variables)

    def _render_round_end(self, rewards: np.array) -> None:
        '''Prints round end information about `rewards`.'''
        print("*********************************************************")
        print(">>> ROUND END")
        print("*********************************************************")
        total_reward = np.sum(rewards)
        print("==> Objective value = {}".format(total_reward))
        print("==> rewards = {}".format(list(rewards)))
        print()
