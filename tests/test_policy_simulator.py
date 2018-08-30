from pyrddl.rddl import RDDL
from pyrddl.parser import RDDLParser

from tfrddlsim.rddl2tf.compiler import Compiler
from tfrddlsim.policy import DefaultPolicy
from tfrddlsim.simulation.policy_simulator import PolicySimulationCell, PolicySimulator

import numpy as np
import tensorflow as tf

import unittest


class TestPolicySimulationCell(unittest.TestCase):

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
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)
        self.batch_size1 = 100
        self.policy1 = DefaultPolicy(self.compiler1, self.batch_size1)
        self.cell1 = PolicySimulationCell(self.compiler1, self.policy1, self.batch_size1)

        self.compiler2 = Compiler(self.rddl2, batch_mode=True)
        self.batch_size2 = 100
        self.policy2 = DefaultPolicy(self.compiler2, self.batch_size2)
        self.cell2 = PolicySimulationCell(self.compiler2, self.policy2, self.batch_size2)

    def test_state_size(self):
        expected = [((8,),), ((3,), (1,), (1,), (1,))]
        cells = [self.cell1, self.cell2]
        for cell, sz in zip(cells, expected):
            state_size = cell.state_size
            self.assertIsInstance(state_size, tuple)
            self.assertTupleEqual(state_size, sz)

    def test_interm_size(self):
        expected = [((8,), (8,), (8,), (8,)), ()]
        cells = [self.cell1, self.cell2]
        for cell, sz in zip(cells, expected):
            interm_size = cell.interm_size
            self.assertIsInstance(interm_size, tuple)
            self.assertTupleEqual(interm_size, sz)

    def test_output_size(self):
        cells = [self.cell1, self.cell2]
        for cell in cells:
            output_size = cell.output_size
            state_size = cell.state_size
            interm_size = cell.interm_size
            action_size = cell.action_size
            self.assertEqual(output_size, (state_size, action_size, interm_size, 1))

    def test_initial_state(self):
        cells = [self.cell1, self.cell2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for cell, batch_size in zip(cells, batch_sizes):
            initial_state = cell.initial_state()
            self.assertIsInstance(initial_state, tuple)
            self.assertEqual(len(initial_state), len(cell.state_size))
            for t, shape in zip(initial_state, cell.state_size):
                self.assertIsInstance(t, tf.Tensor)
                expected_shape = [batch_size] + list(shape)
                if len(expected_shape) == 1:
                    expected_shape += [1]
                self.assertListEqual(t.shape.as_list(), expected_shape)

    def test_simulation_step(self):
        horizon = 40
        cells = [self.cell1, self.cell2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for cell, batch_size in zip(cells, batch_sizes):
            with cell.graph.as_default():
                # initial_state
                initial_state = cell.initial_state()

                # timestep
                timestep = tf.constant(horizon, dtype=tf.float32)
                timestep = tf.expand_dims(timestep, -1)
                timestep = tf.stack([timestep] * batch_size)

                # simulation step
                output, next_state = cell(timestep, initial_state)
                self.assertIsInstance(output, tuple)
                self.assertEqual(len(output), 4)

                next_state, action, interm, reward = output
                state_size, action_size, interm_size, reward_size = cell.output_size

                # interm_state
                # TO DO

                # next_state
                self.assertIsInstance(next_state, tuple)
                self.assertEqual(len(next_state), len(state_size))
                for s, sz in zip(next_state, state_size):
                    self.assertIsInstance(s, tf.Tensor)
                    self.assertListEqual(s.shape.as_list(), [batch_size] + list(sz))

                # action
                self.assertIsInstance(action, tuple)
                self.assertEqual(len(action), len(action_size))
                for a, sz in zip(action, action_size):
                    self.assertIsInstance(a, tf.Tensor)
                    self.assertListEqual(a.shape.as_list(), [batch_size] + list(sz))

                # reward
                self.assertIsInstance(reward, tf.Tensor)
                self.assertListEqual(reward.shape.as_list(), [batch_size, reward_size])


class TestPolicySimulator(unittest.TestCase):

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
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)
        self.batch_size1 = 100
        self.policy1 = DefaultPolicy(self.compiler1, self.batch_size1)
        self.simulator1 = PolicySimulator(self.compiler1, self.policy1, self.batch_size1)

        self.compiler2 = Compiler(self.rddl2, batch_mode=True)
        self.batch_size2 = 100
        self.policy2 = DefaultPolicy(self.compiler2, self.batch_size2)
        self.simulator2 = PolicySimulator(self.compiler2, self.policy2, self.batch_size1)

    def test_timesteps(self):
        horizon = 40
        simulators = [self.simulator1, self.simulator2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for simulator, batch_size in zip(simulators, batch_sizes):
            with simulator.graph.as_default():
                timesteps = simulator.timesteps(horizon)
            self.assertIsInstance(timesteps, tf.Tensor)
            self.assertListEqual(timesteps.shape.as_list(), [batch_size, horizon, 1])
            with tf.Session(graph=simulator.graph) as sess:
                timesteps = sess.run(timesteps)
                for t in timesteps:
                    self.assertListEqual(list(t), list(np.arange(horizon-1, -1, -1)))

    def test_trajectory(self):
        horizon = 40
        compilers = [self.compiler1, self.compiler2]
        simulators = [self.simulator1, self.simulator2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for compiler, simulator, batch_size in zip(compilers, simulators, batch_sizes):
            # trajectory
            trajectory = simulator.trajectory(horizon)
            self.assertIsInstance(trajectory, tuple)
            self.assertEqual(len(trajectory), 5)
            initial_state, states, actions, interms, rewards = trajectory

            # tensor sizes
            state_size, action_size, interm_size, reward_size = simulator.output_size

            # tensor dtypes
            state_dtype = compiler.state_dtype
            interm_dtype = compiler.interm_dtype
            action_dtype = compiler.action_dtype

            # states
            self.assertIsInstance(states, tuple)
            self.assertEqual(len(states), len(state_size))
            for s, sz, dtype in zip(states, state_size, state_dtype):
                self.assertIsInstance(s, tf.Tensor)
                self.assertListEqual(s.shape.as_list(), [batch_size, horizon] + list(sz), '{}'.format(s))
                self.assertEqual(s.dtype, dtype, '{}.dtype != {}'.format(s, dtype))

            # interms
            self.assertIsInstance(interms, tuple)
            self.assertEqual(len(interms), len(interm_size))
            for s, sz, dtype in zip(interms, interm_size, interm_dtype):
                self.assertIsInstance(s, tf.Tensor)
                self.assertListEqual(s.shape.as_list(), [batch_size, horizon] + list(sz), '{}'.format(s))
                self.assertEqual(s.dtype, dtype, '{}.dtype != {}'.format(s, dtype))


            # actions
            self.assertIsInstance(actions, tuple)
            self.assertEqual(len(actions), len(action_size))
            for a, sz, dtype in zip(actions, action_size, action_dtype):
                self.assertIsInstance(a, tf.Tensor)
                self.assertListEqual(a.shape.as_list(), [batch_size, horizon] + list(sz))
                self.assertEqual(a.dtype, dtype)

            # rewards
            self.assertIsInstance(rewards, tf.Tensor)
            self.assertListEqual(rewards.shape.as_list(), [batch_size, horizon, reward_size])

    def test_simulation(self):
        horizon = 40
        compilers = [self.compiler1, self.compiler2]
        simulators = [self.simulator1, self.simulator2]
        batch_sizes = [self.batch_size1, self.batch_size2]
        for compiler, simulator, batch_size in zip(compilers, simulators, batch_sizes):
            # trajectory
            non_fluents, initial_state, states, actions, interms, rewards = simulator.run(horizon)

            # tensor sizes
            state_size, action_size, interm_size, reward_size = simulator.output_size

            # fluent ordering
            state_fluent_ordering = compiler.state_fluent_ordering
            action_fluent_ordering = compiler.action_fluent_ordering

            # states
            self.assertIsInstance(states, tuple)
            self.assertEqual(len(states), len(state_size))
            for name, s, sz in zip(state_fluent_ordering, states, state_size):
                self.assertIsInstance(s, tuple)
                self.assertEqual(len(s), 2)
                var_name, fluent = s
                self.assertIsInstance(var_name, str)
                self.assertEqual(var_name, name, s)
                self.assertIsInstance(fluent, np.ndarray)
                self.assertListEqual(list(fluent.shape), [batch_size, horizon] + list(sz))

            # actions
            self.assertIsInstance(actions, tuple)
            self.assertEqual(len(actions), len(action_size))
            for name, a, sz in zip(action_fluent_ordering, actions, action_size):
                self.assertIsInstance(a, tuple)
                self.assertEqual(len(a), 2)
                var_name, fluent = a
                self.assertIsInstance(var_name, str)
                self.assertEqual(var_name, name, a)
                self.assertIsInstance(fluent, np.ndarray)
                self.assertListEqual(list(fluent.shape), [batch_size, horizon] + list(sz))

            # rewards
            self.assertIsInstance(rewards, np.ndarray)
            self.assertListEqual(list(rewards.shape), [batch_size, horizon])
