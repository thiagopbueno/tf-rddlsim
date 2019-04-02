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


from tfrddlsim.viz.abstract_visualizer import Visualizer
from rddl2tf.compiler import Compiler

import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence, Optional, Tuple, Union
Value = Union[bool, int, float, np.array]
NonFluents = Sequence[Tuple[str, Value]]
Fluents = Sequence[Tuple[str, np.array]]


class NavigationVisualizer(Visualizer):
    '''Visualizer for the Navigation domain.

    It uses Matplotlib.pyplot to render a graphical representation
    of the navigation paths, the initial and goal positions, and
    the deceleration zones.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler
        verbose (bool): Verbosity flag
    '''

    def __init__(self, compiler: Compiler, verbose: bool) -> None:
        super().__init__(compiler, verbose)

    def render(self,
            trajectories: Tuple[NonFluents, Fluents, Fluents, Fluents, np.array],
            batch: Optional[int] = None) -> None:
        '''Render the simulated state-action `trajectories` for Navigation domain.

        Args:
            stats: Performance statistics.
            trajectories: NonFluents, states, actions, interms and rewards.
            batch: Number of batches to render.
        '''

        non_fluents, initial_state, states, actions, interms, rewards = trajectories

        non_fluents = dict(non_fluents)
        states  = dict((name, fluent[0]) for name, fluent in states)
        actions = dict((name, fluent[0]) for name, fluent in actions)
        rewards = rewards[0]

        idx = self._compiler.rddl.domain.state_fluent_ordering.index('location/1')

        start = initial_state[idx][0]
        g = non_fluents['GOAL/1']
        path = states['location/1']
        deltas = actions['move/1']

        centers = non_fluents['DECELERATION_ZONE_CENTER/2']
        decays = non_fluents['DECELERATION_ZONE_DECAY/1']
        zones = [(x, y, d) for (x, y), d in zip(centers, decays)]

        self._ax1 = plt.gca()

        self._render_state_space()
        self._render_start_and_goal_positions(start, g)
        self._render_deceleration_zones(zones)
        self._render_state_action_trajectory(start, path, deltas)

        plt.title('Navigation', fontweight='bold')
        plt.legend(loc='lower right')
        plt.show()

    def _render_state_space(self):
        # lower, upper = (-5.0, -5.0), (10.0, 10.0)
        # self._ax1.axis([lower[0], upper[0], lower[1], upper[1]])
        self._ax1.set_aspect("equal")
        self._ax1.set_xlabel("x coordinate")
        self._ax1.set_ylabel("y coordinate")
        self._ax1.grid()

    def _render_start_and_goal_positions(self, start, goal):
        self._ax1.plot([start[0]], [start[1]], marker='X', markersize=15, color='limegreen', label='initial')
        self._ax1.plot([goal[0]], [goal[1]], marker='X', markersize=15, color='crimson', label='goal')

    def _render_deceleration_zones(self, zones, npoints=1000):
        lower, upper = (-5.0, -5.0), (10.0, 10.0)
        X, Y = np.meshgrid(np.linspace(lower[0], upper[0], npoints), np.linspace(lower[1], upper[1], npoints))
        Lambda = 1.0
        for xcenter, ycenter, decay in zones:
            D = np.sqrt((X - xcenter) ** 2 + (Y - ycenter) ** 2)
            Lambda *= 2 / (1 + np.exp(- decay * D)) - 1.00
        ticks = np.arange(0.0, 1.01, 0.10)
        cp = self._ax1.contourf(X, Y, Lambda, ticks, cmap=plt.cm.bone)
        plt.colorbar(cp, ticks=ticks)
        cp = self._ax1.contour(X, Y, Lambda, ticks, colors="black", linestyles="dashed")

    def _render_state_action_trajectory(self, start, path, deltas):
        xpath = [ p[0] for p in path ]
        ypath = [ p[1] for p in path ]
        self._ax1.plot(xpath, ypath, 'b.', label='states')

        x0, y0 = start
        xdeltas = [ d[0] for d in deltas ]
        ydeltas = [ d[1] for d in deltas ]
        self._ax1.quiver([x0] + xpath[:-1], [y0] + ypath[:-1], xdeltas, ydeltas,
            angles='xy', scale_units='xy', scale=1, color='dodgerblue', width=0.005,
            label='actions')
