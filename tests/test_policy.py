from pyrddl.rddl import RDDL
from pyrddl.parser import RDDLParser

from tfrddlsim.rddl2tf.compiler import Compiler
from tfrddlsim.policy import DefaultPolicy, RandomPolicy

import numpy as np
import tensorflow as tf

import unittest


class TestDefaultPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parser = RDDLParser()
        parser.build()

        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()
            cls.rddl1 = parser.parse(RESERVOIR)
            cls.compiler1 = Compiler(cls.rddl1)

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            MARS_ROVER = file.read()
            cls.rddl2 = parser.parse(MARS_ROVER)
            cls.compiler2 = Compiler(cls.rddl2)

    def test_default_policy(self):
        for compiler in [self.compiler1, self.compiler2]:
            with compiler.graph.as_default():
                default = compiler.default_action_fluents
                batch_size = 1000
                policy = DefaultPolicy(compiler, batch_size)

                state1 = compiler.initial_state_fluents
                action1 = policy(state1, None)
                self.assertIsInstance(action1, tuple)
                for af, (_, t) in zip(action1, default):
                    shape = af.shape.as_list()
                    self.assertEqual(shape[0], batch_size)
                    actual_shape = shape[1:]
                    expected_shape = t.shape.as_list()
                    if expected_shape == []:
                        expected_shape = [1]
                    self.assertListEqual(actual_shape, expected_shape)

                state2 = None
                action2 = policy(state2, None)
                self.assertIs(action1, action2)


class TestRandomPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parser = RDDLParser()
        parser.build()

        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()
            cls.rddl1 = parser.parse(RESERVOIR)
            cls.compiler1 = Compiler(cls.rddl1, batch_mode=True)

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            MARS_ROVER = file.read()
            cls.rddl2 = parser.parse(MARS_ROVER)
            cls.compiler2 = Compiler(cls.rddl2, batch_mode=True)

    def test_random_policy(self):
        batch_size = 1000

        compilers = [self.compiler1, self.compiler2]
        for i, compiler in enumerate(compilers):

            with compiler.graph.as_default():

                policy = RandomPolicy(compiler, batch_size)

                state = compiler.compile_initial_state(batch_size)
                action, n, checking = policy._sample_actions(state)

                action_size = compiler.action_size
                action_dtype = compiler.action_dtype
                action_default_fluents = compiler.compile_default_action(batch_size)

                self.assertIsInstance(action, tuple)
                self.assertEqual(len(action), len(action_size))
                self.assertEqual(len(action), len(action_dtype))
                for fluent, size, dtype, default in zip(action, action_size, action_dtype, action_default_fluents):
                    self.assertIsInstance(fluent, tf.Tensor)
                    self.assertListEqual(fluent.shape.as_list(), [batch_size] + list(size))
                    self.assertEqual(fluent.dtype, dtype)
                    self.assertEqual(fluent.shape, default.shape)
                    self.assertEqual(fluent.dtype, default.dtype)

    def test_random_policy_preconditions_checking(self):
        batch_size = 1000

        compilers = [self.compiler1, self.compiler2]
        for i, compiler in enumerate(compilers):

            with compiler.graph.as_default():

                policy = RandomPolicy(compiler, batch_size)

                state = compiler.compile_initial_state(batch_size)
                action, n, checking = policy._sample_actions(state)

                with tf.Session() as sess:
                    n_, action_, checking_  = sess.run([n, action, checking])
                    self.assertTrue(np.all(checking_))
                    if i == 0: # reservoir: all preconditions are bound constraints
                        self.assertEqual(n_, 0)
                    elif i == 1: # mars rover: xMove and yMove no greater than 5.0
                        xMove, yMove = action_[1:]
                        for dx, dy in zip(xMove.flatten(), yMove.flatten()):
                            self.assertLessEqual(np.abs(dx), RandomPolicy.MAX_REAL_VALUE)
                            self.assertGreaterEqual(np.abs(dy), -RandomPolicy.MAX_REAL_VALUE)
