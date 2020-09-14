import rddlgym

from tfrddlsim.policy import RandomPolicy
from tfrddlsim.simulation.policy_simulator import PolicySimulator
from tfrddlsim.viz import GenericVisualizer

# parse and compile RDDL
compiler = rddlgym.make('Reservoir-8', mode=rddlgym.SCG)
compiler.init()

# run simulations
horizon = 40
batch_size = 128
policy = RandomPolicy(compiler, batch_size)
simulator = PolicySimulator(compiler, policy, batch_size)
trajectories = simulator.run(horizon)

# visualize trajectories
viz = GenericVisualizer(compiler, verbose=True)
viz.render(trajectories)
