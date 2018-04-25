from tfrddlsim.tensorscope import TensorScope

import tensorflow as tf


class TensorFluent(object):

    def __init__(self, tensor, scope):
        self.tensor = tensor
        self.scope = TensorScope(scope)

    @property
    def shape(self):
        return self.tensor.shape.as_list()

    @property
    def dtype(self):
        return self.tensor.dtype

    @classmethod
    def constant(cls, value, dtype=tf.float32):
        t = tf.constant(value, dtype=dtype)
        return TensorFluent(t, [])

    @classmethod
    def Normal(cls, mean, variance):
        if mean.scope != variance.scope:
            raise ValueError('Normal distribution: parameters must have same scope!')
        loc = mean.tensor
        scale = tf.sqrt(variance.tensor)
        t = tf.distributions.Normal(loc, scale).sample()
        scope = mean.scope[:]
        return TensorFluent(t, scope)

    @classmethod
    def Uniform(cls, low, high):
        if low.scope != high.scope:
            raise ValueError('Uniform distribution: parameters must have same scope!')
        low = low.tensor
        high = high.tensor
        t = tf.distributions.Uniform(low, high).sample()
        scope = low.scope
        return TensorFluent(t, scope)

    @classmethod
    def Exponential(cls, mean):
        rate = 1 / mean.tensor
        t = tf.distributions.Exponential(rate).sample()
        scope = mean.scope
        return TensorFluent(t, scope)

    @classmethod
    def Gamma(cls, shape, scale):
        if shape.scope != scale.scope:
            raise ValueError('Gamma distribution: parameters must have same scope!')
        concentration = shape.tensor
        rate = 1 / scale.tensor
        t = tf.distributions.Gamma(concentration, rate).sample()
        scope = shape.scope[:]
        return TensorFluent(t, scope)

    @classmethod
    def abs(cls, x):
        return cls._unary_op(x, tf.abs, tf.float32)

    @classmethod
    def exp(cls, x):
        return cls._unary_op(x, tf.exp, tf.float32)

    @classmethod
    def log(cls, x):
        return cls._unary_op(x, tf.log, tf.float32)

    @classmethod
    def sqrt(cls, x):
        return cls._unary_op(x, tf.sqrt, tf.float32)

    @classmethod
    def cos(cls, x):
        return cls._unary_op(x, tf.cos, tf.float32)

    @classmethod
    def sin(cls, x):
        return cls._unary_op(x, tf.sin, tf.float32)

    @classmethod
    def tan(cls, x):
        return cls._unary_op(x, tf.tan, tf.float32)

    @classmethod
    def round(cls, x):
        return cls._unary_op(x, tf.round, tf.float32)

    @classmethod
    def ceil(cls, x):
        return cls._unary_op(x, tf.ceil, tf.float32)

    @classmethod
    def floor(cls, x):
        return cls._unary_op(x, tf.floor, tf.float32)

    @classmethod
    def pow(cls, x, y):
        return cls._binary_op(x, y, tf.pow, tf.float32)

    @classmethod
    def maximum(cls, x, y):
        return cls._binary_op(x, y, tf.maximum, tf.float32)

    @classmethod
    def minimum(cls, x, y):
        return cls._binary_op(x, y, tf.minimum, tf.float32)

    @classmethod
    def if_then_else(cls, condition, true_case, false_case):
        condition_tensor = condition.tensor
        true_case_tensor = true_case.tensor
        false_case_tensor = false_case.tensor

        if true_case.shape != false_case.shape:
            if true_case.shape == []:
                true_case_tensor = tf.fill(false_case.shape, true_case.tensor)
            elif false_case.shape == []:
                false_case_tensor = tf.fill(true_case.shape, false_case.tensor)

        if true_case_tensor.shape != false_case_tensor.shape:
            raise ValueError('TensorFluent.if_then_else: cases must be of same shape!')

        scope = condition.scope
        t = tf.where(condition_tensor, x=true_case_tensor, y=false_case_tensor)
        return TensorFluent(t, scope[:])

    @classmethod
    def _binary_op(cls, x, y, op, dtype):
        x = x.cast(dtype)
        y = y.cast(dtype)
        s1 = x.scope.as_list()
        s2 = y.scope.as_list()
        scope, perm1, perm2 = TensorScope.broadcast(s1, s2)
        x = x.transpose(perm1)
        y = y.transpose(perm2)
        t = op(x.tensor, y.tensor)
        return TensorFluent(t, scope)

    @classmethod
    def _unary_op(cls, x, op, dtype):
        x = x.cast(dtype)
        t = op(x.tensor)
        scope = x.scope
        return TensorFluent(t, scope[:])

    def cast(self, dtype):
        t = self.tensor if self.tensor.dtype == dtype else tf.cast(self.tensor, dtype)
        scope = self.scope[:]
        return TensorFluent(t, scope)

    def transpose(self, perm=None):
        t = tf.transpose(self.tensor, perm) if perm != [] else self.tensor
        scope = self.scope[:]
        return TensorFluent(t, scope)

    def sum(self, vars_list=None):
        axis = []
        for var in vars_list:
            if var in self.scope:
                axis.append(self.scope.index(var))
        t = tf.reduce_sum(self.tensor, axis=axis)

        scope = []
        for var in self.scope:
            if var not in vars_list:
                scope.append(var)

        return TensorFluent(t, scope)

    def __add__(self, other):
        return self._binary_op(self, other, tf.add, tf.float32)

    def __sub__(self, other):
        return self._binary_op(self, other, tf.subtract, tf.float32)

    def __mul__(self, other):
        return self._binary_op(self, other, tf.multiply, tf.float32)

    def __truediv__(self, other):
        return self._binary_op(self, other, tf.divide, tf.float32)

    def __and__(self, other):
        return self._binary_op(self, other, tf.logical_and, tf.bool)

    def __or__(self, other):
        return self._binary_op(self, other, tf.logical_or, tf.bool)

    def __xor__(self, other):
        return self._binary_op(self, other, tf.logical_xor, tf.bool)

    def __invert__(self):
        return self._unary_op(self, tf.logical_not, tf.bool)

    def __le__(self, other):
        return self._binary_op(self, other, tf.less_equal, tf.float32)

    def __lt__(self, other):
        return self._binary_op(self, other, tf.less, tf.float32)

    def __ge__(self, other):
        return self._binary_op(self, other, tf.greater_equal, tf.float32)

    def __gt__(self, other):
        return self._binary_op(self, other, tf.greater, tf.float32)

    def __eq__(self, other):
        return self._binary_op(self, other, tf.equal, tf.float32)

    def __ne__(self, other):
        return self._binary_op(self, other, tf.not_equal, tf.float32)

    def __str__(self):
        return '{} : {}'.format(self.scope, self.tensor)
