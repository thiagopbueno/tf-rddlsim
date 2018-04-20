class Expression(object):

    def __init__(self, expr):
        self._expr = expr

    def __getitem__(self, i):
        return self._expr[i]

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
