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

from rddl2tf.compiler import Compiler

from tfrddlsim.simulation.transition_simulator import ActionSimulationCell

import tensorflow as tf
import unittest


class TestActionSimulationCell(unittest.TestCase):

    def setUp(self):
        self.batch_size = 10

        self.rddl1 = rddlgym.make('Reservoir-8', mode=rddlgym.AST)
        self.rddl2 = rddlgym.make('Navigation-v2', mode=rddlgym.AST)
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)
        self.compiler2 = Compiler(self.rddl2, batch_mode=True)

        self.cell1 = ActionSimulationCell(self.compiler1)
        self.initial_state1 = self.compiler1.compile_initial_state(batch_size=self.batch_size)
        self.default_action1 = self.compiler1.compile_default_action(batch_size=1)

        self.cell2 = ActionSimulationCell(self.compiler2)
        self.initial_state2 = self.compiler2.compile_initial_state(batch_size=self.batch_size)
        self.default_action2 = self.compiler2.compile_default_action(batch_size=1)

    def test_state_size(self):
        state_size1 = self.cell1.state_size
        self.assertIsInstance(state_size1, tuple)
        self.assertEqual(len(state_size1), len(self.initial_state1))
        for shape, tensor in zip(state_size1, self.initial_state1):
            self.assertListEqual(list(shape), tensor.shape.as_list()[1:])

        state_size2 = self.cell2.state_size
        self.assertIsInstance(state_size2, tuple)
        self.assertEqual(len(state_size2), len(self.initial_state2))
        for shape, tensor in zip(state_size2, self.initial_state2):
            self.assertListEqual(list(shape), tensor.shape.as_list()[1:])

    def test_interm_size(self):
        expected = [((8,), (8,), (8,), (8,)), ((2,), (2,))]
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

    def test_next_state(self):
        cells = [self.cell1, self.cell2]
        actions = [self.default_action1, self.default_action2]
        states = [self.initial_state1, self.initial_state2]
        for cell, inputs, state in zip(cells, actions, states):

            output, next_state = cell(inputs, state)
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
                self.assertListEqual(s.shape.as_list(), [self.batch_size] + list(sz))

            # action
            self.assertIsInstance(action, tuple)
            self.assertEqual(len(action), len(action_size))
            for a, sz in zip(action, action_size):
                self.assertIsInstance(a, tf.Tensor)
                self.assertListEqual(a.shape.as_list(), [1] + list(sz))

            # reward
            self.assertIsInstance(reward, tf.Tensor)
            self.assertListEqual(reward.shape.as_list(), [self.batch_size, reward_size])


    def test_output(self):
        (output1, next_state1) = self.cell1(self.default_action1, self.initial_state1)
        next_state1, action1, interm_state1, reward1 = output1
        self.assertIsInstance(reward1, tf.Tensor)
        self.assertListEqual(reward1.shape.as_list(), [self.batch_size, 1])
        self.assertEqual(reward1.dtype, tf.float32)

        (output2, next_state2) = self.cell2(self.default_action2, self.initial_state2)
        next_state2, action2, interm_state2, reward2 = output2
        self.assertIsInstance(reward2, tf.Tensor)
        self.assertListEqual(reward2.shape.as_list(), [self.batch_size, 1])
        self.assertEqual(reward2.dtype, tf.float32)
