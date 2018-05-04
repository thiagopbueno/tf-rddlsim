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
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_non_fluent() }

    @property
    def state_fluents(self):
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_state_fluent() }

    @property
    def action_fluents(self):
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_action_fluent() }

    @property
    def intermediate_fluents(self):
        return { str(pvar): pvar for pvar in self.pvariables if pvar.is_intermediate_fluent() }

    @property
    def intermediate_cpfs(self):
        _, cpfs = self.cpfs
        interm_cpfs = (cpf for cpf in cpfs if cpf.name in self.intermediate_fluents)
        interm_cpfs = sorted(interm_cpfs, key=lambda cpf: self.intermediate_fluents[cpf.name].level)
        return interm_cpfs

    @property
    def state_cpfs(self):
        _, cpfs = self.cpfs
        state_cpfs = []
        for cpf in cpfs:
            name = cpf.name
            functor = name[:name.index('/')-1]
            arity = name[name.index('/')+1:]
            name = '{}/{}'.format(functor, arity)
            if name in self.state_fluents:
                state_cpfs.append(cpf)
        state_cpfs = sorted(state_cpfs, key=lambda cpf: cpf.name)
        return state_cpfs


class Instance(object):

    def __init__(self, name, sections):
        self.name = name
        self.__dict__.update(sections)


class NonFluents(object):

    def __init__(self, name, sections):
        self.name = name
        self.__dict__.update(sections)
