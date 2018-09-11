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

import abc
import numpy as np

from typing import Sequence, Optional, Tuple, Union
Value = Union[bool, int, float, np.array]
NonFluents = Sequence[Tuple[str, Value]]
Fluents = Sequence[Tuple[str, np.array]]


class Visualizer(metaclass=abc.ABCMeta):
    '''Abstract class for RDDL's trajectory Visualizer.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): RDDL2TensorFlow compiler
        verbose (bool): Verbosity flag
    '''

    def __init__(self, compiler: Compiler, verbose: bool) -> None:
        self._compiler = compiler
        self._verbose = verbose

    @abc.abstractmethod
    def render(self,
            trajectories: Tuple[NonFluents, Fluents, Fluents, Fluents, np.array],
            batch: Optional[int] = None) -> None:
        '''Renders the simulated `trajectories` for the given `batch`.

        Args:
            trajectories: NonFluents, states, actions, interms and rewards.
            batch: Number of batches to render.
        '''
        raise NotImplementedError
