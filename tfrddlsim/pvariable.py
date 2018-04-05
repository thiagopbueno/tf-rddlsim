class NonFluent(object):

    def __init__(self, name, range_type, param_types=None, def_value=None):
        self.name = name
        self.range = range_type
        self.param_types = param_types
        self.def_value = def_value

    def arity(self):
        return len(self.param_types) if self.param_types is not None else 0

    def __repr__(self):
        if self.arity() > 0:
            if self.def_value is not None:
                nf = '{}({}): {{ non-fluent, {}, default = {} }}'.format(self.name, ', '.join(self.param_types), self.range, self.def_value)
            else:
                nf = '{}({}): {{ non-fluent, {} }}'.format(self.name, ', '.join(self.param_types), self.range)
        else:
            if self.def_value is not None:
                nf = '{}: {{ non-fluent, {}, default = {} }}'.format(self.name, self.range, self.def_value)
            else:
                nf = '{}: {{ non-fluent, {} }}'.format(self.name, self.range)
        return nf
