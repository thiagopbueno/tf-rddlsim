class PVariable(object):

    def __init__(self, name, fluent_type, range_type, param_types=None, default=None, level=None):
        self.name = name
        self.fluent_type = fluent_type
        self.range = range_type
        self.param_types = param_types
        self.default = default
        self.level = level

    def arity(self):
        return len(self.param_types) if self.param_types is not None else 0
