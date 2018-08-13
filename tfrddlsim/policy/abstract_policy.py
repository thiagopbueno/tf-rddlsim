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


import abc
import tensorflow as tf

from typing import Sequence


class Policy(metaclass=abc.ABCMeta):
    '''Abstract base class for representing Policy functions.'''

    @abc.abstractmethod
    def __call__(self,
            state: Sequence[tf.Tensor],
            timestep: tf.Tensor) -> Sequence[tf.Tensor]:
        '''Returns action fluents for the current `state` and `timestep`.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            timestep (tf.Tensor): The current timestep.

        Returns:
            Sequence[tf.Tensor]: A tuple of action fluents.
        '''
        raise NotImplementedError
