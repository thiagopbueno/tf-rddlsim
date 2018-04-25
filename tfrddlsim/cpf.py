from tfrddlsim.expr import Expression

class CPF(object):

    def __init__(self, pvar, expr):
        self.pvar = pvar
        self.expr = expr

    @property
    def name(self):
        return Expression._pvar_to_name(self.pvar[1])

    def __repr__(self):
        cpf = '{} = {};'.format(str(self.pvar), str(self.expr))
        return cpf
