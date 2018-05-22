from tfrddlsim.fluent import TensorFluent

import numpy as np
import tensorflow as tf


class Compiler(object):

    def __init__(self, rddl, graph, batch_mode=False):
        self.rddl = rddl
        self.graph = graph
        self.batch_mode = batch_mode

    def batch_mode_on(self):
        self.batch_mode = True

    def batch_mode_off(self):
        self.batch_mode = False

    def compile_initial_state(self, batch_size):
        return self._compile_batch_fluents(self.initial_state_fluents, batch_size)

    def compile_default_action(self, batch_size):
        return self._compile_batch_fluents(self.default_action_fluents, batch_size)

    def compile_cpfs(self, scope):
        next_state_fluents = []

        for cpf in self.rddl.domain.intermediate_cpfs:
            t = self._compile_expression(cpf.expr, scope)
            scope[cpf.name] = t

        for cpf in self.rddl.domain.state_cpfs:
            t = self._compile_expression(cpf.expr, scope)
            next_state_fluents.append((cpf.name, t))

        key = lambda f: self.next_state_fluent_ordering.index(f[0])
        next_state_fluents = sorted(next_state_fluents, key=key)

        return next_state_fluents

    def compile_reward(self, scope):
        reward_expr = self.rddl.domain.reward
        t = self._compile_expression(reward_expr, scope)
        return t

    def non_fluents_scope(self):
        return dict(self.non_fluents)

    def state_scope(self, state_fluents):
        return dict(zip(self.state_fluent_ordering, state_fluents))

    def action_scope(self, action_fluents):
        return dict(zip(self.action_fluent_ordering, action_fluents))

    def next_state_scope(self, next_state_fluents):
        return dict(zip(self.next_state_fluent_ordering, next_state_fluents))

    @property
    def object_table(self):
        if self.__dict__.get('_object_table') is None:
            self._build_object_table()
        return self._object_table

    @property
    def non_fluents(self):
        if self.__dict__.get('_non_fluents') is None:
            self._instantiate_non_fluents()
        return self._non_fluents

    @property
    def initial_state_fluents(self):
        if self.__dict__.get('_initial_state_fluents') is None:
            self._instantiate_initial_state_fluents()
        return self._initial_state_fluents

    @property
    def default_action_fluents(self):
        if self.__dict__.get('_default_action_fluents') is None:
            self._instantiate_default_action_fluents()
        return self._default_action_fluents

    @property
    def local_action_preconditions(self):
        if self.__dict__.get('_local_action_preconditions') is None:
            self._build_preconditions_table()
        return self._local_action_preconditions

    @property
    def global_action_preconditions(self):
        if self.__dict__.get('_global_action_preconditions') is None:
            self._build_preconditions_table()
        return self._global_action_preconditions

    @property
    def non_fluent_ordering(self):
        return [name for name in sorted(self.rddl.domain.non_fluents)]

    @property
    def state_fluent_ordering(self):
        return [name for name in sorted(self.rddl.domain.state_fluents)]

    @property
    def action_fluent_ordering(self):
        return [name for name in sorted(self.rddl.domain.action_fluents)]

    @property
    def next_state_fluent_ordering(self):
        key = lambda x: x.name
        return [cpf.name for cpf in sorted(self.rddl.domain.state_cpfs, key=key)]

    @property
    def state_size(self):
        return self._fluent_size(self.initial_state_fluents, self.state_fluent_ordering)

    @property
    def action_size(self):
        return self._fluent_size(self.default_action_fluents, self.action_fluent_ordering)

    @classmethod
    def _fluent_size(cls, fluents, ordering):
        fluents = dict(fluents)
        size = []
        for name in ordering:
            fluent_shape = fluents[name].shape.fluent_shape
            if fluent_shape == ():
                fluent_shape = (1,)
            size.append(fluent_shape)
        return tuple(size)

    def _build_object_table(self):
        types = self.rddl.domain.types
        objects = dict(self.rddl.non_fluents.objects)
        self._object_table = dict()
        for name, value in self.rddl.domain.types:
            if value == 'object':
                objs = objects[name]
                idx = { obj: i for i, obj in enumerate(objs) }
                self._object_table[name] = { 'size': len(objs), 'idx': idx }

    def _build_preconditions_table(self):
        self._local_action_preconditions = dict()
        self._global_action_preconditions = []
        action_fluents = self.rddl.domain.action_fluents
        for precond in self.rddl.domain.preconds:
            scope = precond.scope
            action_scope = [action for action in scope if action in action_fluents]
            if len(action_scope) == 1:
                name = action_scope[0]
                self._local_action_preconditions[name] = self._local_action_preconditions.get(name, [])
                self._local_action_preconditions[name].append(precond)
            else:
                self._global_action_preconditions.append(precond)

    def _instantiate_pvariables(self, pvariables, ordering, initializer=None):
        if initializer is not None:
            init = dict()
            for ((name, args), value) in initializer:
                arity = len(args) if args is not None else 0
                name = '{}/{}'.format(name, arity)
                init[name] = init.get(name, [])
                init[name].append((args, value))

        fluents = []

        for name in ordering:
            pvar = pvariables[name]
            shape = self._param_types_to_shape(pvar.param_types)
            dtype = self._range_type_to_dtype(pvar.range)
            fluent = np.full(shape, pvar.default)

            if initializer is not None:
                for args, val in init.get(name, []):
                    if args is not None:
                        idx = []
                        for ptype, arg in zip(pvar.param_types, args):
                            idx.append(self.object_table[ptype]['idx'][arg])
                        idx = tuple(idx)
                        fluent[idx] = val
                    else:
                        fluent = val

            with self.graph.as_default():
                t = tf.constant(fluent, dtype=dtype, name=name)
                scope = [None] * len(t.shape)
                fluent = TensorFluent(t, scope, batch=False)
                fluent_pair = (name, fluent)
                fluents.append(fluent_pair)

        return fluents

    def _instantiate_non_fluents(self):
        non_fluents = self.rddl.domain.non_fluents
        initializer = self.rddl.non_fluents.init_non_fluent
        self._non_fluents = self._instantiate_pvariables(non_fluents, self.non_fluent_ordering, initializer)
        return self._non_fluents

    def _instantiate_initial_state_fluents(self):
        state_fluents = self.rddl.domain.state_fluents
        initializer = self.rddl.instance.init_state
        self._initial_state_fluents = self._instantiate_pvariables(state_fluents, self.state_fluent_ordering, initializer)
        return self._initial_state_fluents

    def _instantiate_default_action_fluents(self):
        action_fluents = self.rddl.domain.action_fluents
        self._default_action_fluents = self._instantiate_pvariables(action_fluents, self.action_fluent_ordering)
        return self._default_action_fluents

    def _compile_batch_fluents(self, fluents, batch_size):
        batch_fluents = []
        with self.graph.as_default():
            for name, fluent in fluents:
                t = tf.stack([fluent.tensor] * batch_size, name=name)
                if t.shape.ndims == 1:
                    t = tf.expand_dims(t, -1)
                batch_fluents.append(t)
        return tuple(batch_fluents)

    def _compile_expression(self, expr, scope):
        etype = expr.etype
        args = expr.args

        with self.graph.as_default():

            if etype[0] == 'number':
                return TensorFluent.constant(args)
            elif etype[0] == 'pvar':
                name = expr._pvar_to_name(args)
                if name not in scope:
                    raise ValueError('Variable {} not in scope.'.format(name))
                fluent = scope[name]
                scope = args[1] if args[1] is not None else []
                if isinstance(fluent, TensorFluent):
                    fluent = TensorFluent(fluent.tensor, scope, batch=fluent.batch)
                elif isinstance(fluent, tf.Tensor):
                    fluent = TensorFluent(fluent, scope, batch=self.batch_mode)
                else:
                    raise ValueError('Variable in scope must be TensorFluent-like: {}'.format(fluent))
                return fluent
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
                    vars_list = [var for _, (var, _) in typed_var_list]
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
        shape = tuple(self.object_table[ptype]['size'] for ptype in param_types)
        return shape
