class RDDL(object):

    def __init__(self, domain=None, instance=None, non_fluents=None):
        self.domain = domain
        self.instance = instance
        self.non_fluents = non_fluents

    def add_block(self, block):
        if isinstance(block, Domain):
            self.domain = block
        elif isinstance(block, Instance):
            self.instance = block
        elif isinstance(block, NonFluents):
            self.non_fluents = block
        else:
            msg = "'{}' is not a valid RDDL block.".format(type(block))
            raise ValueError(msg)


class Domain(object):

    def __init__(self, name):
        self.name = name


class Instance(object):

    def __init__(self, name):
        self.name = name


class NonFluents(object):

    def __init__(self, name):
        self.name = name
