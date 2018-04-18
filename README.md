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
