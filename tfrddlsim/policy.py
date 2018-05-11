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
