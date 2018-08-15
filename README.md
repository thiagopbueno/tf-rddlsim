# tf-rddlsim [![Build Status](https://travis-ci.org/thiagopbueno/tf-rddlsim.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-rddlsim) [![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/tf-rddlsim/blob/master/LICENSE)

RDDL2TensorFlow compiler and trajectory simulator in Python3.

# Quickstart

```text
$ pip3 install tfrddlsim
```

# Usage

tf-rddlsim can be used as a standalone script or programmatically.


### Script mode

```text
$ tfrddlsim main.py --help

usage: main.py [-h] [--policy {random,default}] [-hr HORIZON] [-b BATCH_SIZE]
               [-v]
               rddl

RDDL2TensorFlow compiler and simulator

positional arguments:
  rddl                  RDDL filepath

optional arguments:
  -h, --help            show this help message and exit
  --policy {random,default}
                        type of policy (default=random)
  -hr HORIZON, --horizon HORIZON
                        number of timesteps of each trajectory (default=40)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of trajectories in a batch (default=75)
  -v, --verbose         verbosity mode
```


### Programmatic mode

```python
from pyrddl.parser import RDDLParser

from tfrddlsim.rddl2tf.compiler import Compiler
from tfrddlsim.policy import RandomPolicy
from tfrddlsim.simulation.policy_simulator import PolicySimulator
from tfrddlsim.viz import GenericVisualizer

# parse RDDL
parser = RDDLParser()
parser.build()
rddl = parser.parse(rddl_file)

# compile RDDL to TensorFlow computation graph
rddl2tf = Compiler(rddl, batch_mode=True)

# run simulations
horizon = 40
batch_size = 75
policy = RandomPolicy(rddl2tf, batch_size)
simulator = Simulator(rddl2tf, policy, batch_size)
trajectories = simulator.run(horizon)

# visualize trajectories
viz = GenericVisualizer(rddl2tf, verbose=True)
viz.render(trajectories)
```

# Parser

The RDDLParser is automatically built from the RDDL BNF grammar by the PLY (Python Lex-Yacc) package. The parser outputs an Abstract Syntax Tree (AST) given a valid RDDL file as input. Note that not all RDDL2 features are currently being parsed (e.g., matrix and vector operations).


# Compiler

The RDDL2TensorFlow compiler translates an AST representing a RDDL file into a computation graph in TensorFlow.


## Parameterized Variables (pvariables)

Each RDDL fluent is compiled to a ``tfrddlsim.TensorFluent`` after instantiation.

A ``tfrddlsim.TensorFluent`` object wraps a ``tf.Tensor`` object. The arity and the number of objects corresponding to the type of each parameter of a fluent are reflected in a ``tfrddlsim.TensorFluentShape`` object (the rank of a ``tfrddlsim.TensorFluent`` corresponds to the fluent arity and the size of its dimensions corresponds to the number of objects of each type). Also, a ``tfrddlsim.TensorFluentShape`` manages batch sizes when evaluating operations in batch mode.

Additionally, a ``tfrddlsim.TensorFluent``keeps information about the ordering of the fluent parameters in a ``tfrddlsim.TensorScope`` object.

The ``tfrddlsim.TensorFluent`` abstraction is necessary in the evaluation of RDDL expressions due the broadcasting rules of operations in TensorFlow.


## Conditional Probability Functions (CPFs)

Each CPF expression is compiled into an operation in a ``tf.Graph``, possibly composed of many other operations. Typical RDDL operations, functions, and probability distributions are mapped to equivalent TensorFlow ops. These operations are added to a ``tf.Graph`` by recursively compiling the expressions in a CPF into wrapped operations and functions implemented at the ``tfrddlsim.TensorFluent`` level.

Note that the RDDL2TensorFlow compiler currently only supports element-wise operations (e.g. ``a(?x, ?y) = b(?x) * c(?y)`` is not allowed). However, all compiled operations are vectorized, i.e., computations are done simultaneously for all object instantiations of a pvariable.

Optionally, during simulation operations can be evaluated in batch mode. In this case, state-action trajectories are generated in parallel by the ``tfrddlsim.Simulator``.


# Simulator

The ``tfrddlsim.Simulator`` implements a stochastic Recurrent Neural Net (RNN) in order to sample state-action trajectories. Each RNN cell encapsulates a ``tfrddlsim.Policy`` module generating actions for current states and comprehends the transition (specified by the CPFs) and reward functions. Sampling is done through dynamic unrolling of the RNN model with the embedded ``tfrddlsim.Policy``.

Note that the ``tfrddlsim`` package only provides a ``tfrddlsim.RandomPolicy`` and a ``tfrddlsim.DefaultPolicy`` (constant policy with all action fluents with default values).


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
