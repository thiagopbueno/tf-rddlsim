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


from tfrddlsim.fluentscope import TensorFluentScope
from tfrddlsim.fluentshape import TensorFluentShape

import tensorflow as tf


class TensorFluent(object):

    def __init__(self, tensor, scope, batch=False):
        self.tensor = tensor
        self.scope = TensorFluentScope(scope)
        self.shape = TensorFluentShape(tensor.shape, batch)

    @property
    def batch(self):
        return self.shape.batch

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def name(self):
        return self.tensor.name

    @classmethod
    def constant(cls, value, dtype=tf.float32):
        t = tf.constant(value, dtype=dtype)
        scope = []
        batch = False
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Normal(cls, mean, variance, batch_size=None):
        if mean.scope != variance.scope:
            raise ValueError('Normal distribution: parameters must have same scope!')
        loc = mean.tensor
        scale = tf.sqrt(variance.tensor)
        dist = tf.distributions.Normal(loc, scale)
        batch = mean.batch or variance.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = mean.scope[:]
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Uniform(cls, low, high, batch_size=None):
        if low.scope != high.scope:
            raise ValueError('Uniform distribution: parameters must have same scope!')
        low = low.tensor
        high = high.tensor
        dist = tf.distributions.Uniform(low, high)
        batch = low.batch or high.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = low.scope
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Exponential(cls, mean, batch_size=None):
        rate = 1 / mean.tensor
        dist = tf.distributions.Exponential(rate)
        batch = mean.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = mean.scope
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def Gamma(cls, shape, scale, batch_size=None):
        if shape.scope != scale.scope:
            raise ValueError('Gamma distribution: parameters must have same scope!')
        concentration = shape.tensor
        rate = 1 / scale.tensor
        dist = tf.distributions.Gamma(concentration, rate)
        batch = shape.batch or scale.batch
        if not batch and batch_size is not None:
            t = dist.sample(batch_size)
            batch = True
        else:
            t = dist.sample()
        scope = shape.scope[:]
        return TensorFluent(t, scope, batch=batch)

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
            if true_case.shape.as_list() == []:
                true_case_tensor = tf.fill(false_case.shape.as_list(), true_case.tensor)
            elif false_case.shape.as_list() == []:
                false_case_tensor = tf.fill(true_case.shape.as_list(), false_case.tensor)

        if true_case_tensor.shape != false_case_tensor.shape:
            raise ValueError('TensorFluent.if_then_else: cases must be of same shape!')

        t = tf.where(condition_tensor, x=true_case_tensor, y=false_case_tensor)
        scope = condition.scope[:]

        batch = condition.batch
        # if (not batch) and (condition.batch or true_case.batch or false_case.batch):
        #     raise ValueError('TensorFluent.if_then_else: cases must be batch compatible!')

        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def _binary_op(cls, x, y, op, dtype):

        # scope
        s1 = x.scope.as_list()
        s2 = y.scope.as_list()
        scope, perm1, perm2 = TensorFluentScope.broadcast(s1, s2)
        if x.batch and perm1 != []:
            perm1 = [0] + [p+1 for p in perm1]
        if y.batch and perm2 != []:
            perm2 = [0] + [p+1 for p in perm2]
        x = x.transpose(perm1)
        y = y.transpose(perm2)

        # shape
        reshape1, reshape2 = TensorFluentShape.broadcast(x.shape, y.shape)
        if reshape1 is not None:
            x = x.reshape(reshape1)
        if reshape2 is not None:
            y = y.reshape(reshape2)

        # dtype
        x = x.cast(dtype)
        y = y.cast(dtype)

        # operation
        t = op(x.tensor, y.tensor)

        # batch
        batch = x.batch or y.batch

        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def _unary_op(cls, x, op, dtype):
        x = x.cast(dtype)
        t = op(x.tensor)
        scope = x.scope[:]
        batch = x.batch
        return TensorFluent(t, scope, batch=batch)

    @classmethod
    def _aggregation_op(cls, op, x, vars_list=None):
        axis = []
        for var in vars_list:
            if var in x.scope:
                ax = x.scope.index(var)
                if x.batch:
                    ax += 1
                axis.append(ax)
        t = op(x.tensor, axis=axis)

        scope = []
        for var in x.scope:
            if var not in vars_list:
                scope.append(var)

        batch = x.batch

        return TensorFluent(t, scope, batch=batch)

    def cast(self, dtype):
        t = self.tensor if self.tensor.dtype == dtype else tf.cast(self.tensor, dtype)
        scope = self.scope[:]
        batch = self.batch
        return TensorFluent(t, scope, batch=batch)

    def reshape(self, shape):
        t = tf.reshape(self.tensor, shape)
        scope = self.scope[:]
        batch = self.batch
        return TensorFluent(t, scope, batch=batch)

    def transpose(self, perm=None):
        t = tf.transpose(self.tensor, perm) if perm != [] else self.tensor
        scope = self.scope[:]
        batch = self.batch
        return TensorFluent(t, scope, batch=batch)

    def sum(self, vars_list=None):
        return self._aggregation_op(tf.reduce_sum, self, vars_list)

    def prod(self, vars_list=None):
        return self._aggregation_op(tf.reduce_prod, self, vars_list)

    def forall(self, vars_list=None):
        return self._aggregation_op(tf.reduce_all, self, vars_list)

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
        return 'TensorFluent("{}", dtype={}, {}, {})'.format(self.name, repr(self.dtype), self.scope, self.shape)
