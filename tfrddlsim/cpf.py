class CPF(object):

    def __init__(self, pvar, expr):
        self.pvar = pvar
        self.expr = expr

    def __repr__(self):
        cpf = '{} = {};'.format(str(self.pvar), str(self.expr))
        return cpf