from tfrddlsim.rddl import RDDL
from tfrddlsim.parser import RDDLParser
from tfrddlsim.compiler import Compiler
from tfrddlsim.policy import DefaultPolicy

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
            cls.graph1 = tf.Graph()
            cls.compiler1 = Compiler(cls.rddl1, cls.graph1)

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            MARS_ROVER = file.read()
            cls.rddl2 = parser.parse(MARS_ROVER)
            cls.graph2 = tf.Graph()
            cls.compiler2 = Compiler(cls.rddl2, cls.graph2)

    def test_policy(self):
        for compiler in [self.compiler1, self.compiler2]:
            with compiler.graph.as_default():
                default = compiler.default_action_fluents
                batch_size = 1000
                policy = DefaultPolicy(compiler, batch_size)

                state1 = compiler.initial_state_fluents
                action1 = policy(state1)
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
                action2 = policy(state2)
                self.assertIs(action1, action2)
