from tfrddlsim.parser     import RDDLParser
from tfrddlsim.compiler   import Compiler
from tfrddlsim.policy     import DefaultPolicy, RandomPolicy
from tfrddlsim.simulator  import Simulator
from tfrddlsim.visualizer import BasicVisualizer


import argparse


def parse_args():
    description = 'RDDL2TensorFlow compiler and simulator'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('rddl', type=str, help='RDDL filepath')
    parser.add_argument(
        '--policy',
        default='random',
        choices=('random', 'default'),
        help='type of policy (default=random)'
    )
    parser.add_argument(
        '-hr', '--horizon',
        type=int, default=40,
        help='number of timesteps of each trajectory (default=40)'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int, default=75,
        help='number of trajectories in a batch (default=75)'
    )
    parser.add_argument('-v', '--verbose', help='verbosity mode')
    return parser.parse_args()


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def compile(rddl):
    # parse RDDL
    parser = RDDLParser()
    parser.build()
    rddl = parser.parse(rddl)

    # build RDDL2TensorFlow compiler
    return Compiler(rddl, batch_mode=True)


def get_policy(compiler, policy_type, batch_size):
    policy = RandomPolicy if policy_type == 'random' else DefaultPolicy
    return policy(compiler, batch_size)


def simulate(compiler, policy, horizon, batch_size):
    simulator = Simulator(compiler, policy, batch_size)
    return simulator.run(horizon)


def display(compiler, trajectories, verbose):
    viz = BasicVisualizer(compiler, verbose)
    viz.render(trajectories)


if __name__ == '__main__':

    # parser CLI arguments
    args = parse_args()

    # read RDDL file
    rddl = read_file(args.rddl)

    # compile RDDL to TensorFlow
    rddl2tf = compile(rddl)

    # build policy
    policy = get_policy(rddl2tf, args.policy, args.batch_size)

    # run simulations
    trajectories = simulate(rddl2tf, policy, args.horizon, args.batch_size)

    # visualize trajectories
    display(rddl2tf, trajectories, args.verbose)
