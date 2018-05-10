import tensorflow as tf

class TensorFluentShape(object):

    def __init__(self, shape, batch):
        self._shape = tf.TensorShape(shape)
        self._batch = batch

    def as_list(self):
        return self._shape.as_list()

    def __getitem__(self, i):
        return self._shape[i]

    def __eq__(self, other):
        return self._shape == other._shape and self._batch == other._batch

    def __ne__(self, other):
        return self._shape != other._shape or self._batch != other._batch

    def __str__(self):
        return 'TensorFluentShape({}, batch={})'.format(self.as_list(), self._batch)

    @property
    def batch(self):
        return self._batch

    @property
    def batch_size(self):
        return self._shape.as_list()[:1] if self._batch else []

    @property
    def fluent_shape(self):
        return self._shape.as_list()[1:] if self._batch else self._shape.as_list()[:]

    @property
    def fluent_size(self):
        return len(self.fluent_shape)

    @classmethod
    def broadcast(cls, shape1, shape2):
        reshape_1, reshape_2 = None, None

        if not (shape1._batch or shape2._batch):
            return reshape_1, reshape_2

        size_1, size_2 = shape1.fluent_size, shape2.fluent_size
        size_diff = abs(size_1 - size_2)
        if size_diff == 0:
            return reshape_1, reshape_2

        if size_2 > size_1 and not (size_1 == 0 and not shape1._batch):
            reshape_1 = shape1.batch_size + [1] * size_diff + shape1.fluent_shape
        elif size_1 > size_2 and not (size_2 == 0 and not shape2._batch):
            reshape_2 = shape2.batch_size + [1] * size_diff + shape2.fluent_shape
        return reshape_1, reshape_2
