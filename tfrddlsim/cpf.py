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


from tfrddlsim.expr import Expression

class CPF(object):

    def __init__(self, pvar, expr):
        self.pvar = pvar
        self.expr = expr

    @property
    def name(self):
        return Expression._pvar_to_name(self.pvar[1])

    def __repr__(self):
        cpf = '{} =\n{};'.format(str(self.pvar), str(self.expr))
        return cpf
