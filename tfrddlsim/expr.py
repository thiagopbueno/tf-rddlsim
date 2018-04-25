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
        elif self._expr[0] == 'sum':
            return self._expr[1]
        elif self._expr[0] == 'abs':
            return self._expr[1]
        elif self._expr[0] == 'if':
            return self._expr[1]

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
