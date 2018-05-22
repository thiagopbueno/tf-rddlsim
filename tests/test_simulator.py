from tfrddlsim.rddl import RDDL
from tfrddlsim.parser import RDDLParser
from tfrddlsim.compiler import Compiler
from tfrddlsim.policy import DefaultPolicy
from tfrddlsim.simulator import SimulationCell, Simulator

import numpy as np
import tensorflow as tf

import unittest


class TestSimulationCell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parser = RDDLParser()
        parser.build()

        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()
            cls.rddl1 = parser.parse(RESERVOIR)

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            MARS_ROVER = file.read()
            cls.rddl2 = parser.parse(MARS_ROVER)

    def setUp(self):
        self.graph1 = tf.Graph()
        self.compiler1 = Compiler(self.rddl1, self.graph1, batch_mode=True)
        self.batch_size1 = 100
        self.policy1 = DefaultPolicy(self.compiler1, self.batch_size1)
        self.cell1 = SimulationCell(self.compiler1, self.policy1, self.batch_size1)

        self.graph2 = tf.Graph()
        self.compiler2 = Compiler(self.rddl2, self.graph2, batch_mode=True)
        self.batch_size2 = 100
        self.policy2 = DefaultPolicy(self.compiler2, self.batch_size2)
        self.cell2 = SimulationCell(self.compiler2, self.policy2, self.batch_size2)

    def test_state_size(self):
        expected = [((8,),), ((3,), (), (), ())]
        cells = [self.cell1, self.cell2]
        for cell, sz in zip(cells, expected):
            state_size = cell.state_size
            self.assertIsInstance(state_size, tuple)
            self.assertTupleEqual(state_size, sz)

    def test_output_size(self):
        cells = [self.cell1, self.cell2]
        for cell in cells:
            output_size = cell.output_size
            self.assertEqual(output_size, 1)

    def test_initial_state(self):
        cells = [self.cell1, self.cell2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for cell, batch_size in zip(cells, batch_sizes):
            initial_state = cell.initial_state()
            self.assertIsInstance(initial_state, tuple)
            self.assertEqual(len(initial_state), len(cell.state_size))
            for t, shape in zip(initial_state, cell.state_size):
                self.assertIsInstance(t, tf.Tensor)
                self.assertListEqual(t.shape.as_list(), [batch_size] + list(shape))


class TestSimulator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parser = RDDLParser()
        parser.build()

        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()
            cls.rddl1 = parser.parse(RESERVOIR)

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            MARS_ROVER = file.read()
            cls.rddl2 = parser.parse(MARS_ROVER)

    def setUp(self):
        self.graph1 = tf.Graph()
        self.compiler1 = Compiler(self.rddl1, self.graph1, batch_mode=True)
        self.batch_size1 = 100
        self.policy1 = DefaultPolicy(self.compiler1, self.batch_size1)
        self.simulator1 = Simulator(self.compiler1, self.policy1, self.batch_size1)

        self.graph2 = tf.Graph()
        self.compiler2 = Compiler(self.rddl2, self.graph2, batch_mode=True)
        self.batch_size2 = 100
        self.policy2 = DefaultPolicy(self.compiler2, self.batch_size2)
        self.simulator2 = Simulator(self.compiler2, self.policy2, self.batch_size1)

    def test_timesteps(self):
        simulators = [self.simulator1, self.simulator2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for simulator, batch_size in zip(simulators, batch_sizes):
            horizon = 40
            with simulator.graph.as_default():
                timesteps = simulator.timesteps(horizon)
            self.assertIsInstance(timesteps, tf.Tensor)
            self.assertListEqual(timesteps.shape.as_list(), [batch_size, horizon, 1])
            with tf.Session(graph=simulator.graph) as sess:
                timesteps = sess.run(timesteps)
                for t in timesteps:
                    self.assertListEqual(list(t), list(np.arange(horizon-1, -1, -1)))

    def test_initial_state(self):
        simulators = [self.simulator1, self.simulator2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        compilers = [self.compiler1, self.compiler2]
        for simulator, batch_size, compiler in zip(simulators, batch_sizes, compilers):
            initial_state = compiler.compile_initial_state(batch_size)
            self.assertIsInstance(initial_state, tuple)
            self.assertEqual(len(initial_state), len(compiler.state_size))
            for fluent, fluent_size in zip(initial_state, compiler.state_size):
                self.assertEqual(fluent.shape[0], batch_size)
                self.assertEqual(fluent.shape[1:], fluent_size)

    def test_trajectory(self):
        horizon = 40
        simulators = [self.simulator1, self.simulator2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for simulator, batch_size in zip(simulators, batch_sizes):
            outputs = simulator.trajectory(horizon, batch_size)
            output_size = simulator.output_size
            self.assertIsInstance(outputs, tf.Tensor)
            self.assertListEqual(outputs.shape.as_list(), [batch_size, horizon, output_size])
