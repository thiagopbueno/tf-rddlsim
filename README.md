# tf-rddlsim [![Build Status](https://travis-ci.org/thiagopbueno/tf-rddlsim.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-rddlsim) [![Documentation Status](https://readthedocs.org/projects/tf-rddlsim/badge/?version=latest)](https://tf-rddlsim.readthedocs.io/en/latest/?badge=latest) [![License](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/thiagopbueno/tf-rddlsim/blob/master/LICENSE)

RDDL2TensorFlow compiler and trajectory simulator in Python3.

# Quickstart

```text
$ pip3 install tfrddlsim
```

# Usage

tf-rddlsim can be used as a standalone script or programmatically.


## Script mode

```text
$ tfrddlsim --help

usage: tfrddlsim [-h] [--policy {default,random}] [--viz {generic,navigation}]
                 [-hr HORIZON] [-b BATCH_SIZE] [-v]
                 rddl

RDDL2TensorFlow compiler and simulator

positional arguments:
  rddl                  path to RDDL file or rddlgym problem id

optional arguments:
  -h, --help            show this help message and exit
  --policy {default,random}
                        type of policy (default=random)
  --viz {generic,navigation}
                        type of visualizer (default=generic)
  -hr HORIZON, --horizon HORIZON
                        number of timesteps of each trajectory (default=40)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        number of trajectories in a batch (default=75)
  -v, --verbose         verbosity mode
```

```text
$ tfrddlsim Navigation-v1 --policy random --viz navigation -hr 50 -b 32 -v
```

```text
$ tfrddlsim Reservoir-8 --policy default --viz generic -hr 20 -b 128 -v
```


## Programmatic mode

```python
import rddlgym

from tfrddlsim.policy import RandomPolicy
from tfrddlsim.simulation.policy_simulator import PolicySimulator
from tfrddlsim.viz import GenericVisualizer

# parse and compile RDDL
rddl2tf = rddlgym.make('Reservoir-8')
rddl2tf.batch_mode_on()

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


# Simulator

The ``tfrddlsim.Simulator`` implements a stochastic Recurrent Neural Net (RNN) in order to sample state-action trajectories. Each RNN cell encapsulates a ``tfrddlsim.Policy`` module generating actions for current states and comprehends the transition (specified by the CPFs) and reward functions. Sampling is done through dynamic unrolling of the RNN model with the embedded ``tfrddlsim.Policy``.

Note that the ``tfrddlsim`` package only provides a ``tfrddlsim.RandomPolicy`` and a ``tfrddlsim.DefaultPolicy`` (constant policy with all action fluents with default values).


# Documentation

Please refer to [https://tf-rddlsim.readthedocs.io/](https://tf-rddlsim.readthedocs.io/en/latest/) for the code documentation.


# Support

If you are having issues with ``tf-rddlsim``, please let me know at: [thiago.pbueno@gmail.com](mailto://thiago.pbueno@gmail.com).


# License

Copyright (c) 2018-2019 Thiago Pereira Bueno All Rights Reserved.

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
