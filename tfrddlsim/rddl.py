# This file is part of tf-rddlsim.

# tf-rddlsim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-rddlsim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-rddlsim. If not, see <http://www.gnu.org/licenses/>.


class RDDL(object):

    def __init__(self, blocks):
        self.__dict__.update(blocks)

    @classmethod
    def rename_next_state_fluent(cls, name):
        i = name.index('/')
        functor = name[:i-1]
        arity = name[i+1:]
        return "{}/{}".format(functor, arity)

    @classmethod
    def rename_state_fluent(cls, name):
        i = name.index('/')
        functor = name[:i]
        arity = name[i+1:]
        return "{}'/{}".format(functor, arity)


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
        interm_cpfs = sorted(interm_cpfs, key=lambda cpf: (self.intermediate_fluents[cpf.name].level, cpf.name))
        return interm_cpfs

    @property
    def state_cpfs(self):
        _, cpfs = self.cpfs
        state_cpfs = []
        for cpf in cpfs:
            name = RDDL.rename_next_state_fluent(cpf.name)
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
