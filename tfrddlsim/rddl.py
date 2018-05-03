class RDDL(object):

    def __init__(self, blocks):
        self.__dict__.update(blocks)


class Domain(object):

    def __init__(self, name, requirements, sections):
        self.name = name
        self.requirements = requirements
        self.__dict__.update(sections)

    @property
    def non_fluents(self):
        return (pvar for pvar in self.pvariables if pvar.is_non_fluent())

    @property
    def state_fluents(self):
        return (pvar for pvar in self.pvariables if pvar.is_state_fluent())

    @property
    def action_fluents(self):
        return (pvar for pvar in self.pvariables if pvar.is_action_fluent())

    @property
    def intermediate_fluents(self):
        return (pvar for pvar in self.pvariables if pvar.is_intermediate_fluent())


class Instance(object):

    def __init__(self, name, sections):
        self.name = name
        self.__dict__.update(sections)


class NonFluents(object):

    def __init__(self, name, sections):
        self.name = name
        self.__dict__.update(sections)
