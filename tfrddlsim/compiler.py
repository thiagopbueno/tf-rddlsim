import numpy as np
import tensorflow as tf


class Compiler(object):

    def __init__(self, rddl, graph):
        self._rddl = rddl
        self._graph = graph

    def _build_object_table(self):
        types = self._rddl.domain.types
        objects = dict(self._rddl.non_fluents.objects)
        self._object_table = dict()
        for name, value in self._rddl.domain.types:
            if value == 'object':
                objs = objects[name]
                idx = { obj: i for i, obj in enumerate(objs) }
                self._object_table[name] = { 'size': len(objs), 'idx': idx }

    def _instantiate_non_fluents(self):
        init_non_fluent = dict()
        for ((name, params), value) in self._rddl.non_fluents.init_non_fluent:
            name = '{}/{}'.format(name, len(params))
            init_non_fluent[name] = init_non_fluent.get(name, [])
            init_non_fluent[name].append((params, value))

        self._non_fluents = dict()
        pvariables = self._rddl.domain.pvariables
        for pvar in pvariables:
            if pvar.is_non_fluent():
                name = str(pvar)
                dtype = self._range_type_to_dtype(pvar.range)
                shape = self._param_types_to_shape(pvar.param_types)
                nf = np.full(shape, pvar.default)

                init = init_non_fluent.get(name, [])
                for args, val in init:
                    idx = []
                    for ptype, arg in zip(pvar.param_types, args):
                        idx.append(self._object_table[ptype]['idx'][arg])
                    idx = tuple(idx)
                    nf[idx] = val

                with self._graph.as_default():
                    constant = tf.constant(nf, dtype=dtype, name=name)
                    self._non_fluents[name] = constant

        return self._non_fluents

    def _instantiate_initial_state_fluents(self):
        state_fluents_initializer = dict()
        for ((name, params), value) in self._rddl.instance.init_state:
            name = '{}/{}'.format(name, len(params))
            state_fluents_initializer[name] = state_fluents_initializer.get(name, [])
            state_fluents_initializer[name].append((params, value))

        self._initial_state_fluents = dict()
        pvariables = self._rddl.domain.pvariables
        for pvar in pvariables:
            if pvar.is_state_fluent():
                name = str(pvar)
                dtype = self._range_type_to_dtype(pvar.range)
                shape = self._param_types_to_shape(pvar.param_types)
                sf = np.full(shape, pvar.default)

                init = state_fluents_initializer.get(name, [])
                for args, val in init:
                    idx = []
                    for ptype, arg in zip(pvar.param_types, args):
                        idx.append(self._object_table[ptype]['idx'][arg])
                    idx = tuple(idx)
                    sf[idx] = val

                with self._graph.as_default():
                    constant = tf.constant(sf, dtype=dtype, name=name)
                    self._initial_state_fluents[name] = constant

        return self._initial_state_fluents

    @classmethod
    def _range_type_to_dtype(cls, range_type):
        dtype = None
        if range_type == 'real':
            dtype = tf.float32
        elif range_type == 'int':
            dtype = tf.int32
        elif range_type == 'bool':
            dtype = tf.bool
        return dtype

    def _param_types_to_shape(self, param_types):
        param_types = [] if param_types is None else param_types
        shape = tuple(self._object_table[ptype]['size'] for ptype in param_types)
        return shape
