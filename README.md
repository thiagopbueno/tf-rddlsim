# tf-rddlsim [![Build Status](https://travis-ci.org/thiagopbueno/tf-rddlsim.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-rddlsim) [![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/tf-mdp/blob/master/LICENSE)

RDDL2TensorFlow compiler and MDP simulator in Python3.

# Usage

```python
# read RDDL file
filename = sys.argv[1]
with open(filename, 'r') as f:
    rddl = f.read()

# build parser
parser = RDDLParser()
parser.build()

# parse RDDL
rddl = parser.parse(rddl)
domain = rddl.domain
non_fluents = rddl.non_fluents
instance = rddl.instance

# compile RDDL to TensorFlow
# TODO
```

# License

Copyright (c) 2018 Thiago Pereira Bueno All Rights Reserved.

tf-rddlsim is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

tf-rddlsim is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tf-rddlsim. If not, see http://www.gnu.org/licenses/.
