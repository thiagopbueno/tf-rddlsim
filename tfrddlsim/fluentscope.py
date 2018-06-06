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


class TensorFluentScope(object):

    def __init__(self, scope):
        self._scope = scope

    def as_list(self):
        return self._scope[:]

    def index(self, i):
        return self._scope.index(i)

    def __len__(self):
        return len(self._scope)

    def __getitem__(self, i):
        return self._scope[i]

    def __eq__(self, other):
        return self._scope == other._scope

    def __ne__(self, other):
        return self._scope != other._scope

    def __str__(self):
        return 'TensorFluentScope({})'.format(str(self._scope))

    @classmethod
    def broadcast(cls, s1, s2):
        if len(s1) == 0:
            return s2, [], []
        if len(s2) == 0:
            return s1, [], []

        subscope = list(set(s1) & set(s2))
        if len(subscope) == len(s1):
            subscope = s1
        elif len(subscope) == len(s2):
            subscope = s2

        perm1 = []
        if s1[-len(subscope):] != subscope:
            i = 0
            for var in s1:
                if var not in subscope:
                    perm1.append(i)
                    i += 1
                else:
                    j = subscope.index(var)
                    perm1.append(len(s1) - len(subscope) + j)
        perm2 = []
        if s2[-len(subscope):] != subscope:
            i = 0
            for var in s2:
                if var not in subscope:
                    perm2.append(i)
                    i += 1
                else:
                    j = subscope.index(var)
                    perm2.append(len(s2) - len(subscope) + j)

        scope = []
        if len(s1) >= len(s2):
            if perm1 == []:
                scope = s1
            else:
                for i in range(len(s1)):
                    scope.append(s1[perm1.index(i)])
        else:
            if perm2 == []:
                scope = s2
            else:
                for i in range(len(s2)):
                    scope.append(s2[perm2.index(i)])

        return scope, perm1, perm2
