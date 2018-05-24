import pprint


class Expression(object):

    def __init__(self, expr):
        self._expr = expr

    def __getitem__(self, i):
        return self._expr[i]

    @property
    def etype(self):
        if self._expr[0] == 'number':
            return ('number', type(self._expr[1]))
        elif self._expr[0] == 'pvar_expr':
            return ('pvar', self._expr[1][0])
        elif self._expr[0] == 'randomvar':
            return ('randomvar', self._expr[1][0])
        elif self._expr[0] in ['+', '-', '*', '/']:
            return ('arithmetic', self._expr[0])
        elif self._expr[0] in ['^', '&', '|', '~', '=>', '<=>']:
            return ('boolean', self._expr[0])
        elif self._expr[0] in ['>=', '<=', '<', '>', '==', '~=']:
            return ('relational', self._expr[0])
        elif self._expr[0] == 'func':
            return ('func', self._expr[1][0])
        elif self._expr[0] == 'sum':
            return ('aggregation', 'sum')
        elif self._expr[0] == 'prod':
            return ('aggregation', 'prod')
        elif self._expr[0] == 'forall':
            return ('aggregation', 'forall')
        elif self._expr[0] == 'exists':
            return ('aggregation', 'exists')
        elif self._expr[0] == 'if':
            return ('control', 'if')

    @property
    def args(self):
        if self._expr[0] == 'number':
            return self._expr[1]
        elif self._expr[0] == 'pvar_expr':
            return self._expr[1]
        elif self._expr[0] == 'randomvar':
            return self._expr[1][1]
        elif self._expr[0] in ['+', '-', '*', '/']:
            return self._expr[1]
        elif self._expr[0] in ['^', '&', '|', '~', '=>', '<=>']:
            return self._expr[1]
        elif self._expr[0] in ['>=', '<=', '<', '>', '==', '~=']:
            return self._expr[1]
        elif self._expr[0] == 'func':
            return self._expr[1][1]
        elif self._expr[0] in ['sum', 'prod', 'forall', 'exists']:
            return self._expr[1]
        elif self._expr[0] == 'abs':
            return self._expr[1]
        elif self._expr[0] == 'if':
            return self._expr[1]

    def is_pvariable_expression(self):
        return self.etype[0] == 'pvar'

    @property
    def name(self):
        if not self.is_pvariable_expression():
            raise ValueError('Expression is not a pvariable.')
        return self._pvar_to_name(self.args)

    def is_number_expression(self):
        return self.etype[0] == 'number'

    @property
    def value(self):
        if not self.is_number_expression():
            raise ValueError('Expression is not a number.')
        return self.args

    def __str__(self):
        return self.__expr_str(self, 0)

    @classmethod
    def __expr_str(cls, expr, level):
        ident = ' ' * level * 4

        if expr.etype[0] in ['pvar', 'number']:
            return '{}Expression(etype={}, args={})'.format(ident, expr.etype, expr.args)

        if not isinstance(expr, Expression):
            return '{}{}'.format(ident, str(expr))

        args = list(cls.__expr_str(arg, level + 1) for arg in expr.args)
        args = '\n'.join(args)
        return '{}Expression(etype={}, args=\n{})'.format(ident, expr.etype, args)

    @property
    def scope(self):
        return self.__get_scope(self._expr)

    @classmethod
    def __get_scope(cls, expr):
        scope = set()
        for i, atom in enumerate(expr):
            if isinstance(atom, Expression):
                scope.update(cls.__get_scope(atom._expr))
            elif type(atom) in [tuple, list]:
                scope.update(cls.__get_scope(atom))
            elif atom == 'pvar_expr':
                functor, params = expr[i+1]
                arity = len(params) if params is not None else 0
                name = '{}/{}'.format(functor, arity)
                scope.add(name)
                break
        return scope

    @classmethod
    def _pvar_to_name(cls, pvar_expr):
        functor = pvar_expr[0]
        arity = len(pvar_expr[1]) if pvar_expr[1] is not None else 0
        return '{}/{}'.format(functor, arity)
