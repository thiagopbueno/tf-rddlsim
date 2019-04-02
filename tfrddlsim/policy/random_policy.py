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

from tfrddlsim.policy.abstract_policy import Policy


import tensorflow as tf

from typing import Dict, Optional, Sequence, Tuple
Constraints = Tuple[Optional[TensorFluent], Optional[TensorFluent]]


class RandomPolicy(Policy):
    '''RandomPolicy class.

    The random policy samples action fluents uniformly. It checks for
    all action preconditions and constraints. The range of each action
    fluent is defined by action bounds constraints if defined in the
    RDDL model, or by default maximum values.
    values.

    Args:
        compiler (:obj:`rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        batch_size (int): The batch size.

    Attributes:
        compiler (:obj:`rddl2tf.compiler.Compiler`): A RDDL2TensorFlow compiler.
        batch_size (int): The batch size.
    '''

    MAX_REAL_VALUE = 5.0
    MAX_INT_VALUE = 5

    def __init__(self, compiler: Compiler, batch_size: int) -> None:
        self.compiler = compiler
        self.batch_size = batch_size

    def __call__(self,
            state: Sequence[tf.Tensor],
            timestep: tf.Tensor) -> Sequence[tf.Tensor]:
        '''Returns sampled action fluents for the current `state` and `timestep`.

        Args:
            state (Sequence[tf.Tensor]): The current state fluents.
            timestep (tf.Tensor): The current timestep.

        Returns:
            Sequence[tf.Tensor]: A tuple of action fluents.
        '''
        action, _, _ = self._sample_actions(state)
        return action

    def _sample_actions(self,
            state: Sequence[tf.Tensor]) -> Tuple[Sequence[tf.Tensor], tf.Tensor, tf.Tensor]:
        '''Returns sampled action fluents and tensors related to the sampling.

        Args:
            state (Sequence[tf.Tensor]): A list of state fluents.

        Returns:
            Tuple[Sequence[tf.Tensor], tf.Tensor, tf.Tensor]: A tuple with
            action fluents, an integer tensor for the number of samples, and
            a boolean tensor for checking all action preconditions.
        '''
        default = self.compiler.compile_default_action(self.batch_size)
        bound_constraints = self.compiler.compile_action_bound_constraints(state)
        action = self._sample_action(bound_constraints, default)
        n, action, checking = self._check_preconditions(state, action, bound_constraints, default)
        return action, n, checking

    def _check_preconditions(self,
            state: Sequence[tf.Tensor],
            action: Sequence[tf.Tensor],
            bound_constraints: Dict[str, Constraints],
            default: Sequence[tf.Tensor]) -> Tuple[tf.Tensor, Sequence[tf.Tensor], tf.Tensor]:
        '''Samples action fluents until all preconditions are satisfied.

        Checks action preconditions for the sampled `action` and current `state`,
        and iff all preconditions are satisfied it returns the sampled action fluents.

        Args:
            state (Sequence[tf.Tensor]): A list of state fluents.
            action (Sequence[tf.Tensor]): A list of action fluents.
            bound_constraints (Dict[str, Tuple[Optional[TensorFluent], Optional[TensorFluent]]]): The bounds for each action fluent.
            default (Sequence[tf.Tensor]): The default action fluents.

        Returns:
            Tuple[tf.Tensor, Sequence[tf.Tensor], tf.Tensor]: A tuple with
            an integer tensor corresponding to the number of samples,
            action fluents and a boolean tensor for checking all action preconditions.
        '''

        def condition(i, a, checking):
            not_checking = tf.reduce_any(tf.logical_not(checking))
            return not_checking

        def body(i, a, checking):
            new_action = []
            new_sampled_action = self._sample_action(bound_constraints, default)
            new_preconds_checking = self.compiler.compile_action_preconditions_checking(state, new_sampled_action)
            for action_fluent, new_sampled_action_fluent in zip(a, new_sampled_action):
                new_action_fluent = tf.where(checking, action_fluent, new_sampled_action_fluent)
                new_action.append(new_action_fluent)
            new_action = tuple(new_action)
            new_checking = tf.logical_or(checking, new_preconds_checking)
            return (i + 1, new_action, new_checking)

        i0 = tf.constant(0)
        preconds_checking = self.compiler.compile_action_preconditions_checking(state, action)
        return tf.while_loop(condition, body, loop_vars=[i0, action, preconds_checking])

    def _sample_action(self,
            constraints: Dict[str, Constraints],
            default: Sequence[tf.Tensor],
            prob: float = 0.3) -> Sequence[tf.Tensor]:
        '''Samples action fluents respecting the given bound `constraints`.

        With probability `prob` it chooses the action fluent default value,
        with probability 1-`prob` it samples the fluent w.r.t. its bounds.

        Args:
            constraints (Dict[str, Tuple[Optional[TensorFluent], Optional[TensorFluent]]]): The bounds for each action fluent.
            default (Sequence[tf.Tensor]): The default action fluents.
            prob (float): A probability measure.

        Returns:
            Sequence[tf.Tensor]: A tuple of action fluents.
        '''
        ordering = self.compiler.rddl.domain.action_fluent_ordering
        dtypes = map(rddl2tf.utils.range_type_to_dtype, self.compiler.rddl.action_range_type)
        size = self.compiler.rddl.action_size

        action = []
        for name, dtype, size, default_value in zip(ordering, dtypes, size, default):
            action_fluent = self._sample_action_fluent(name, dtype, size, constraints, default_value, prob)
            action.append(action_fluent)

        return tuple(action)

    def _sample_action_fluent(self,
            name: str,
            dtype: tf.DType,
            size: Sequence[int],
            constraints: Dict[str, Constraints],
            default_value: tf.Tensor,
            prob: float) -> tf.Tensor:
        '''Samples the action fluent with given `name`, `dtype`, and `size`.

        With probability `prob` it chooses the action fluent `default_value`,
        with probability 1-`prob` it samples the fluent w.r.t. its `constraints`.

        Args:
            name (str): The name of the action fluent.
            dtype (tf.DType): The data type of the action fluent.
            size (Sequence[int]): The size and shape of the action fluent.
            constraints (Dict[str, Tuple[Optional[TensorFluent], Optional[TensorFluent]]]): The bounds for each action fluent.
            default_value (tf.Tensor): The default value for the action fluent.
            prob (float): A probability measure.

        Returns:
            tf.Tensor: A tensor for sampling the action fluent.
        '''
        shape = [self.batch_size] + list(size)

        if dtype == tf.float32:
            bounds = constraints.get(name)
            if bounds is None:
                low, high = -self.MAX_REAL_VALUE, self.MAX_REAL_VALUE
                dist = tf.distributions.Uniform(low=low, high=high)
                sampled_fluent = dist.sample(shape)
            else:
                low, high = bounds
                batch = (low is not None and low.batch) or (high is not None and high.batch)
                low = tf.cast(low.tensor, tf.float32) if low is not None else -self.MAX_REAL_VALUE
                high = tf.cast(high.tensor, tf.float32) if high is not None else self.MAX_REAL_VALUE
                dist = tf.distributions.Uniform(low=low, high=high)
                if batch:
                    sampled_fluent = dist.sample()
                elif isinstance(low, tf.Tensor) or isinstance(high, tf.Tensor):
                    if (low+high).shape.as_list() == list(size):
                        sampled_fluent = dist.sample([self.batch_size])
                    else:
                        raise ValueError('bounds are not compatible with action fluent.')
                else:
                    sampled_fluent = dist.sample(shape)
        elif dtype == tf.int32:
            logits = [1.0] * self.MAX_INT_VALUE
            dist = tf.distributions.Categorical(logits=logits, dtype=tf.int32)
            sampled_fluent = dist.sample(shape)
        elif dtype == tf.bool:
            probs = 0.5
            dist = tf.distributions.Bernoulli(probs=probs, dtype=tf.bool)
            sampled_fluent = dist.sample(shape)

        select_default = tf.distributions.Bernoulli(prob, dtype=tf.bool).sample(self.batch_size)
        action_fluent = tf.where(select_default, default_value, sampled_fluent)

        return action_fluent
