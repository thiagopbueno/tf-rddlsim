import rddlgym
from rddl2tf.compilers import DefaultCompiler as Compiler
from tfrddlsim.policy import RandomPolicy
from tfrddlsim.simulation.policy_simulator import PolicySimulator
from tfrddlsim.viz import GenericVisualizer

# parameters
horizon = 40
batch_size = 32

# parse and compile RDDL
rddl = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
compiler = Compiler(rddl, batch_size)
compiler.init()

# run simulations
policy = RandomPolicy(compiler)
simulator = PolicySimulator(compiler, policy)
trajectories = simulator.run(horizon)

# visualize trajectories
viz = GenericVisualizer(compiler, verbose=True)
viz.render(trajectories)
