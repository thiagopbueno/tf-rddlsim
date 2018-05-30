# tf-rddlsim [![Build Status](https://travis-ci.org/thiagopbueno/tf-rddlsim.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-rddlsim) [![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/tf-mdp/blob/master/LICENSE)

RDDL2TensorFlow compiler and MDP simulator in Python3.

# Usage

```bash
$ python3 main.py rddl/Reservoir.rddl
```

```python
# read RDDL file
filename = sys.argv[1]
with open(filename, 'r') as f:
    rddl = f.read()

# parse RDDL
parser = RDDLParser()
parser.build()
rddl = parser.parse(rddl)

# compile RDDL to TensorFlow
graph = tf.Graph()
compiler = Compiler(rddl, graph, batch_mode=True)

# run simulations
horizon = 40
batch_size = 75
policy = DefaultPolicy(compiler, batch_size)
simulator = Simulator(compiler, policy, batch_size)
trajectories = simulator.run(horizon)

# visualize trajectories
viz = BasicVisualizer(compiler, verbose=2)
viz.render(trajectories)
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
