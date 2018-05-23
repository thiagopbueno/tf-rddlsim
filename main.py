from tfrddlsim.parser     import RDDLParser
from tfrddlsim.compiler   import Compiler
from tfrddlsim.policy     import DefaultPolicy
from tfrddlsim.simulator  import Simulator
from tfrddlsim.visualizer import BasicVisualizer

import sys
import tensorflow as tf


if __name__ == '__main__':

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
