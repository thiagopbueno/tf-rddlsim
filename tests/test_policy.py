# This file is part of tf-rddlsim.

# tf-rddlsim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# tf-rddlsim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with tf-rddlsim. If not, see <http://www.gnu.org/licenses/>.


import rddlgym

import rddl2tf
from rddl2tf.compiler import Compiler

from tfrddlsim.policy import DefaultPolicy, RandomPolicy

import numpy as np
import tensorflow as tf

import unittest


class TestDefaultPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        cls.rddl2 = rddlgym.make('Mars_Rover', mode=rddlgym.AST)
        cls.compiler1 = Compiler(cls.rddl1)
        cls.compiler2 = Compiler(cls.rddl2)

    def test_default_policy(self):
        for compiler in [self.compiler1, self.compiler2]:
            with compiler.graph.as_default():
                default = compiler.compile_default_action()
                batch_size = 1000
                policy = DefaultPolicy(compiler, batch_size)

                state1 = compiler.compile_initial_state()
                action1 = policy(state1, None)
                self.assertIsInstance(action1, tuple)
                for af, (_, t) in zip(action1, default):
                    shape = af.shape.as_list()
                    self.assertEqual(shape[0], batch_size)
                    actual_shape = shape[1:]
                    expected_shape = t.shape.as_list()
                    self.assertListEqual(actual_shape, expected_shape)

                state2 = None
                action2 = policy(state2, None)
                self.assertIs(action1, action2)


class TestRandomPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        cls.rddl2 = rddlgym.make('Mars_Rover', mode=rddlgym.AST)
        cls.compiler1 = Compiler(cls.rddl1, batch_mode=True)
        cls.compiler2 = Compiler(cls.rddl2, batch_mode=True)

    def test_random_policy(self):
        batch_size = 1000

        compilers = [self.compiler1, self.compiler2]
        for i, compiler in enumerate(compilers):

            with compiler.graph.as_default():

                policy = RandomPolicy(compiler, batch_size)

                state = compiler.compile_initial_state(batch_size)
                action, n, checking = policy._sample_actions(state)

                action_size = compiler.rddl.action_size
                action_range_type = compiler.rddl.action_range_type
                action_default_fluents = compiler.compile_default_action(batch_size)

                self.assertIsInstance(action, tuple)
                self.assertEqual(len(action), len(action_size))
                self.assertEqual(len(action), len(action_range_type))
                for fluent, size, range_type, default in zip(action, action_size, action_range_type, action_default_fluents):
                    self.assertIsInstance(fluent, tf.Tensor)
                    self.assertListEqual(fluent.shape.as_list(), [batch_size] + list(size))
                    self.assertEqual(fluent.dtype, rddl2tf.utils.range_type_to_dtype(range_type))
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
