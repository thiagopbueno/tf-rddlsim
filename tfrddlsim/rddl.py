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

    @property
    def pvariables(self):
        return self.domain_list.get('pvariables')

    @property
    def cpfs(self):
        return self.domain_list.get('cpfs')

    @property
    def cdfs(self):
        return self.domain_list.get('cdfs')

    @property
    def reward(self):
        return self.domain_list.get('reward')

    @property
    def preconds(self):
        return self.domain_list.get('preconds')

    @property
    def constraints(self):
        return self.domain_list.get('constraints')

    @property
    def invariants(self):
        return self.domain_list.get('invariants')


class Instance(object):

    def __init__(self, name):
        self.name = name


class NonFluents(object):

    def __init__(self, name):
        self.name = name
