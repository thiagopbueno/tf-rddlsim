class RDDL(object):

    def __init__(self, blocks):
        self.blocks = blocks

    @property
    def domain(self):
        return self.blocks.get('domain')

    @property
    def instance(self):
        return self.blocks.get('instance')

    @property
    def non_fluents(self):
        return self.blocks.get('non_fluents')


class Domain(object):

    def __init__(self, name, requirements, domain_list):
        self.name = name
        self.requirements = requirements
        self.domain_list = domain_list

    @property
    def types(self):
        return self.domain_list.get('types')


class Instance(object):

    def __init__(self, name):
        self.name = name


class NonFluents(object):

    def __init__(self, name):
        self.name = name
