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

from pyrddl.parser import RDDLParser

from tfrddlsim.compiler import Compiler
from tfrddlsim.simulation.transition_simulator import ActionSimulationCell

import tensorflow as tf
import unittest


class TestActionSimulationCell(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        parser = RDDLParser()
        parser.build()

        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()
            cls.rddl1 = parser.parse(RESERVOIR)

        with open('rddl/Navigation.rddl', mode='r') as file:
            NAVIGATION = file.read()
            cls.rddl2 = parser.parse(NAVIGATION)

    def setUp(self):
        self.compiler1 = Compiler(self.rddl1, batch_mode=True)
        self.cell1 = ActionSimulationCell(self.compiler1)
        self.initial_state1 = self.compiler1.compile_initial_state(batch_size=10)
        self.default_action1 = self.compiler1.compile_default_action(batch_size=1)

        self.compiler2 = Compiler(self.rddl2, batch_mode=True)
        self.cell2 = ActionSimulationCell(self.compiler2)
        self.initial_state2 = self.compiler2.compile_initial_state(batch_size=10)
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

    def test_next_state(self):
        (output1, next_state1) = self.cell1(self.initial_state1, self.default_action1)
        print(next_state1)
        self.assertIsInstance(next_state1, tuple)
        self.assertEqual(len(next_state1), len(self.initial_state1))
        for t1, t2 in zip(self.initial_state1, next_state1):
            self.assertIsInstance(t2, tf.Tensor)
            self.assertEqual(t1.shape, t2.shape)
            self.assertEqual(t1.dtype, t2.dtype)

        (output2, next_state2) = self.cell2(self.initial_state2, self.default_action2)
        print(next_state2)
        self.assertIsInstance(next_state2, tuple)
        self.assertEqual(len(next_state2), len(self.initial_state2))
        for t1, t2 in zip(self.initial_state2, next_state2):
            self.assertIsInstance(t2, tf.Tensor)
            self.assertEqual(t1.shape, t2.shape)
            self.assertEqual(t1.dtype, t2.dtype)

    @unittest.skip('not implemented until RDDL2TF outputs transition probabilities')
    def test_output_size(self):
        self.fail()

    @unittest.skip('not implemented until RDDL2TF outputs transition probabilities')
    def test_output(self):
        self.fail()
