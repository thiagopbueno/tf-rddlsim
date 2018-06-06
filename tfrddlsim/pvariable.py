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


class PVariable(object):

    def __init__(self, name, fluent_type, range_type, param_types=None, default=None, level=None):
        self.name = name
        self.fluent_type = fluent_type
        self.range = range_type
        self.param_types = param_types
        self.default = default
        self.level = level

    @property
    def arity(self):
        return len(self.param_types) if self.param_types is not None else 0

    def is_non_fluent(self):
        return self.fluent_type == 'non-fluent'

    def is_state_fluent(self):
        return self.fluent_type == 'state-fluent'

    def is_action_fluent(self):
        return self.fluent_type == 'action-fluent'

    def is_intermediate_fluent(self):
        return self.fluent_type == 'interm-fluent'

    def __str__(self):
        return '{}/{}'.format(self.name, self.arity)

    def __repr__(self):
        return self.name if self.arity == 0 else '{}({})'.format(self.name, ','.join(self.param_types))
