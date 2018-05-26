import abc
import tensorflow as tf


class Policy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, state, timestep=None):
        return


class DefaultPolicy(Policy):

    def __init__(self, compiler, batch_size):
        self._default = compiler.compile_default_action(batch_size)

    def __call__(self, state, timestep=None):
        return self._default


class RandomPolicy(Policy):

    MAX_REAL_VALUE = 5.0
    MAX_INT_VALUE = 5

    def __init__(self, compiler, batch_size):
        self.compiler = compiler
        self.batch_size = batch_size

    def __call__(self, state, timestep=None):
        action, _, _ = self._sample_actions(state)
        return action

    def _sample_actions(self, state):
        default = self.compiler.compile_default_action(self.batch_size)
        bound_constraints = self.compiler.compile_action_bound_constraints(state)
        action = self._sample_action(bound_constraints, default)
        n, action, checking = self._check_preconditions(state, action, bound_constraints, default)
        return action, n, checking

    def _check_preconditions(self, state, action, bound_constraints, default):

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

    def _sample_action(self, constraints, default, prob=0.3):
        ordering = self.compiler.action_fluent_ordering
        dtype = self.compiler.action_dtype
        size = self.compiler.action_size

        action = []
        for name, dtype, size, default_value in zip(ordering, dtype, size, default):
            action_fluent = self._sample_action_fluent(name, dtype, size, constraints, default_value, prob)
            action.append(action_fluent)

        return tuple(action)

    def _sample_action_fluent(self, name, dtype, size, constraints, default_value, prob):
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
                low = low.tensor if low is not None else -self.MAX_REAL_VALUE
                high = high.tensor if high is not None else self.MAX_REAL_VALUE
                dist = tf.distributions.Uniform(low=low, high=high)
                if batch:
                    sampled_fluent = dist.sample()
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
