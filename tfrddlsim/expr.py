class Expression(object):

    def __init__(self, expr):
        self._expr = expr

    def __getitem__(self, i):
        return self._expr[i]

    @property
    def scope(self):
        scope = set()
        for i, atom in enumerate(self._expr):
            if type(atom) in [tuple, list]:
                scope.update(self._get_scope(atom))
            elif atom == 'pvar_expr':
                functor, params = self._expr[i+1]
                arity = len(params) if params is not None else 0
                name = '{}/{}'.format(functor, arity)
                scope.add(name)
                break
        return scope
