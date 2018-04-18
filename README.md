# tf-rddlsim

RDDL2TensorFlow compiler and MDP simulator

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
