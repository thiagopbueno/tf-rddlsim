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
        action = []

        action_fluent_ordering = self.compiler.action_fluent_ordering
        action_dtype = self.compiler.action_dtype
        action_size = self.compiler.action_size

        action_bound_constraints = self.compiler.compile_action_bound_constraints(state)

        for name, dtype, size in zip(action_fluent_ordering, action_dtype, action_size):
            shape = [self.batch_size] + list(size)

            if dtype == tf.float32:
                bounds = action_bound_constraints.get(name)
                if bounds is None:
                    low, high = -self.MAX_REAL_VALUE, self.MAX_REAL_VALUE
                    dist = tf.distributions.Uniform(low=low, high=high)
                    a = dist.sample(shape)
                else:
                    low, high = bounds
                    batch = (low is not None and low.batch) or (high is not None and high.batch)
                    low = low.tensor if low is not None else -self.MAX_REAL_VALUE
                    high = high.tensor if high is not None else self.MAX_REAL_VALUE
                    dist = tf.distributions.Uniform(low=low, high=high)
                    if batch:
                        a = dist.sample()
                    else:
                        a = dist.sample(shape)
            elif dtype == tf.int32:
                logits = [1.0] * self.MAX_INT_VALUE
                dist = tf.distributions.Categorical(logits=logits, dtype=tf.int32)
                a = dist.sample(shape)
            elif dtype == tf.bool:
                probs = 0.5
                dist = tf.distributions.Bernoulli(probs=probs, dtype=tf.bool)
                a = dist.sample(shape)

            action.append(a)

        return tuple(action)
