from tfrddlsim.tensorfluent import TensorFluent

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

    def _build_pvariable_table(self):
        self._pvariable_table = {
            'non_fluents': {},
            'state_fluents': {},
            'action_fluents': {},
            'intermediate_fluents': {}
        }
        for pvar in self._rddl.domain.pvariables:
            name = str(pvar)
            if pvar.is_non_fluent():
                self._pvariable_table['non_fluents'][name] = pvar
            elif pvar.is_state_fluent():
                self._pvariable_table['state_fluents'][name] = pvar
            elif pvar.is_action_fluent():
                self._pvariable_table['action_fluents'][name] = pvar
            elif pvar.is_intermediate_fluent():
                self._pvariable_table['intermediate_fluents'][name] = pvar

    def _build_preconditions_table(self):
        self._local_action_preconditions = dict()
        self._global_action_preconditions = []
        action_fluents = self._pvariable_table['action_fluents']
        for precond in self._rddl.domain.preconds:
            scope = precond.scope
            action_scope = [action for action in scope if action in action_fluents]
            if len(action_scope) == 1:
                name = action_scope[0]
                self._local_action_preconditions[name] = self._local_action_preconditions.get(name, [])
                self._local_action_preconditions[name].append(precond)
            else:
                self._global_action_preconditions.append(precond)

    def _instantiate_pvariables(self, pvariables, initializer=None):

        if initializer is not None:
            init = dict()
            for ((name, params), value) in initializer:
                arity = len(params) if params is not None else 0
                name = '{}/{}'.format(name, arity)
                init[name] = init.get(name, [])
                init[name].append((params, value))

        fluents = {}

        for pvar in pvariables:
            name = str(pvar)
            shape = self._param_types_to_shape(pvar.param_types)
            dtype = self._range_type_to_dtype(pvar.range)
            fluent = np.full(shape, pvar.default)

            if initializer is not None:
                for args, val in init.get(name, []):
                    if args is not None:
                        idx = []
                        for ptype, arg in zip(pvar.param_types, args):
                            idx.append(self._object_table[ptype]['idx'][arg])
                        idx = tuple(idx)
                        fluent[idx] = val
                    else:
                        fluent = val

            with self._graph.as_default():
                fluents[name] = tf.constant(fluent, dtype=dtype, name=name)

        return fluents

    def _instantiate_non_fluents(self):
        non_fluents = self._rddl.domain.non_fluents
        initializer = self._rddl.non_fluents.init_non_fluent
        self._non_fluents = self._instantiate_pvariables(non_fluents, initializer)
        return self._non_fluents

    def _instantiate_initial_state_fluents(self):
        state_fluents = self._rddl.domain.state_fluents
        initializer = self._rddl.instance.init_state
        self._initial_state_fluents = self._instantiate_pvariables(state_fluents, initializer)
        return self._initial_state_fluents

    def _instantiate_default_action_fluents(self):
        action_fluents = self._rddl.domain.action_fluents
        self._default_action_fluents = self._instantiate_pvariables(action_fluents)
        return self._default_action_fluents

    def _compile_expression(self, expr, scope):
        etype = expr.etype
        args = expr.args

        with self._graph.as_default():

            if etype[0] == 'number':
                return TensorFluent.constant(args)
            elif etype[0] == 'pvar':
                name = expr._pvar_to_name(args)
                if name not in scope:
                    raise ValueError('Variable {} not in scope.'.format(name))
                t = scope[name]
                s = args[1] if args[1] is not None else []
                return TensorFluent(t, s)
            elif etype[0] == 'randomvar':
                if etype[1] == 'Normal':
                    mean = self._compile_expression(args[0], scope)
                    variance = self._compile_expression(args[1], scope)
                    return TensorFluent.Normal(mean, variance)
                elif etype[1] == 'Uniform':
                    low = self._compile_expression(args[0], scope)
                    high = self._compile_expression(args[1], scope)
                    return TensorFluent.Uniform(low, high)
                elif etype[1] == 'Exponential':
                    mean = self._compile_expression(args[0], scope)
                    return TensorFluent.Exponential(mean)
                elif etype[1] == 'Gamma':
                    shape = self._compile_expression(args[0], scope)
                    scale = self._compile_expression(args[1], scope)
                    return TensorFluent.Gamma(shape, scale)
            elif etype[0] == 'arithmetic':
                if etype[1] == '+':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 + op2
                elif etype[1] == '-':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 - op2
                elif etype[1] == '*':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 * op2
                elif etype[1] == '/':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 / op2
            elif etype[0] == 'boolean':
                if etype[1] in ['^', '&']:
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 & op2
                elif etype[1] == '|':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 | op2
                elif etype[1] == '=>':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return ~op1 | op2
                elif etype[1] == '<=>':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return (op1 & op2) | (~op1 & ~op2)
                elif etype[1] == '~':
                    op = self._compile_expression(args[0], scope)
                    return ~op
            elif etype[0] == 'relational':
                if etype[1] == '<=':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 <= op2
                elif etype[1] == '<':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 < op2
                elif etype[1] == '>=':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 >= op2
                elif etype[1] == '>':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 > op2
                elif etype[1] == '==':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 == op2
                elif etype[1] == '~=':
                    op1 = self._compile_expression(args[0], scope)
                    op2 = self._compile_expression(args[1], scope)
                    return op1 != op2
            elif etype[0] == 'func':
                if etype[1] == 'abs':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.abs(x)
                elif etype[1] == 'exp':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.exp(x)
                elif etype[1] == 'log':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.log(x)
                elif etype[1] == 'sqrt':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.sqrt(x)
                elif etype[1] == 'cos':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.cos(x)
                elif etype[1] == 'sin':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.sin(x)
                elif etype[1] == 'round':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.round(x)
                elif etype[1] == 'ceil':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.ceil(x)
                elif etype[1] == 'floor':
                    x = self._compile_expression(args[0], scope)
                    return TensorFluent.floor(x)
                elif etype[1] == 'pow':
                    x = self._compile_expression(args[0], scope)
                    y = self._compile_expression(args[1], scope)
                    return TensorFluent.pow(x, y)
                elif etype[1] == 'max':
                    x = self._compile_expression(args[0], scope)
                    y = self._compile_expression(args[1], scope)
                    return TensorFluent.maximum(x, y)
                elif etype[1] == 'min':
                    x = self._compile_expression(args[0], scope)
                    y = self._compile_expression(args[1], scope)
                    return TensorFluent.minimum(x, y)
            elif etype[0] == 'control':
                if etype[1] == 'if':
                    condition = self._compile_expression(args[0], scope)
                    true_case = self._compile_expression(args[1], scope)
                    false_case = self._compile_expression(args[2], scope)
                    return TensorFluent.if_then_else(condition, true_case, false_case)
            elif etype[0] == 'aggregation':
                if etype[1] == 'sum':
                    typed_var_list = args[:-1]
                    vars_list = [var for _, (var, _) in typed_var_list ]
                    expr = args[-1]
                    op = self._compile_expression(expr, scope)
                    return op.sum(vars_list=vars_list)

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
