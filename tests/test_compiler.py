from tfrddlsim.parser import RDDLParser
from tfrddlsim.compiler import Compiler
from tfrddlsim.tensorfluent import TensorFluent
from tfrddlsim.expr import Expression

import numpy as np
import tensorflow as tf

import unittest


class TestCompiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            MARS_ROVER = file.read()

        parser = RDDLParser()
        parser.build()

        cls.rddl1 = parser.parse(RESERVOIR)
        cls.rddl2 = parser.parse(MARS_ROVER)

    def setUp(self):
        self.graph1 = tf.Graph()
        self.compiler1 = Compiler(self.rddl1, self.graph1)
        self.assertIs(self.compiler1._rddl, self.rddl1)
        self.assertIs(self.compiler1._graph, self.graph1)

        self.graph2 = tf.Graph()
        self.compiler2 = Compiler(self.rddl2, self.graph2)
        self.assertIs(self.compiler2._rddl, self.rddl2)
        self.assertIs(self.compiler2._graph, self.graph2)

    def test_build_object_table(self):
        self.compiler1._build_object_table()
        self.assertIn('res', self.compiler1._object_table)
        size = self.compiler1._object_table['res']['size']
        idx = self.compiler1._object_table['res']['idx']
        self.assertEqual(size, 8)
        objs = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
        for i, obj in enumerate(objs):
            self.assertIn(obj, idx)
            self.assertEqual(idx[obj], i)

    def test_build_pvariable_table(self):
        self.compiler1._build_pvariable_table()

        expected = {
            'non_fluents': {
                'MAX_RES_CAP/1',
                'UPPER_BOUND/1',
                'LOWER_BOUND/1',
                'RAIN_SHAPE/1',
                'RAIN_SCALE/1',
                'DOWNSTREAM/2',
                'SINK_RES/1',
                'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0',
                'LOW_PENALTY/1',
                'HIGH_PENALTY/1'
            },
            'intermediate_fluents': {
                'evaporated/1',
                'rainfall/1',
                'overflow/1'
            },
            'state_fluents': {
                'rlevel/1'
            },
            'action_fluents': {
                'outflow/1'
            }
        }

        self.assertIsInstance(self.compiler1._pvariable_table, dict)
        for fluent_type, fluents in self.compiler1._pvariable_table.items():
            self.assertIn(fluent_type, expected)
            self.assertSetEqual(set(fluents), expected[fluent_type])

    def test_build_action_preconditions_table(self):
        self.compiler1._build_pvariable_table()
        self.compiler1._build_preconditions_table()

        local_preconds = self.compiler1._local_action_preconditions
        self.assertIsInstance(local_preconds, dict)
        self.assertEqual(len(local_preconds), 1)
        self.assertIn('outflow/1', local_preconds)
        self.assertEqual(len(local_preconds['outflow/1']), 2)

        global_preconds = self.compiler1._global_action_preconditions
        self.assertIsInstance(global_preconds, list)
        self.assertEqual(len(global_preconds), 0)

    def test_instantiate_non_fluents(self):
        self.compiler1._build_object_table()
        nf = self.compiler1._instantiate_non_fluents()

        expected_non_fluents = {
            'MAX_RES_CAP/1': { 'shape': (8,), 'dtype': tf.float32 },
            'UPPER_BOUND/1': { 'shape': (8,), 'dtype': tf.float32 },
            'LOWER_BOUND/1': { 'shape': (8,), 'dtype': tf.float32 },
            'RAIN_SHAPE/1': { 'shape': (8,), 'dtype': tf.float32 },
            'RAIN_SCALE/1': { 'shape': (8,), 'dtype': tf.float32 },
            'DOWNSTREAM/2': { 'shape': (8,8), 'dtype': tf.bool },
            'SINK_RES/1': { 'shape': (8,), 'dtype': tf.bool },
            'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0': { 'shape': (), 'dtype': tf.float32 },
            'LOW_PENALTY/1': { 'shape': (8,), 'dtype': tf.float32 },
            'HIGH_PENALTY/1': { 'shape': (8,), 'dtype': tf.float32 }
        }
        self.assertIsInstance(nf, dict)
        self.assertEqual(len(nf), len(expected_non_fluents))
        for name, tensor in nf.items():
            self.assertIn(name, expected_non_fluents)
            shape = expected_non_fluents[name]['shape']
            dtype = expected_non_fluents[name]['dtype']
            self.assertEqual(tensor.name, '{}:0'.format(name))
            self.assertIsInstance(tensor, tf.Tensor)
            self.assertEqual(tensor.dtype, dtype)
            self.assertEqual(tensor.shape, shape)

        expected_initializers = {
            'MAX_RES_CAP/1': [ 100.,  100.,  200.,  300.,  400.,  500.,  800., 1000.],
            'UPPER_BOUND/1': [ 80.,  80., 180., 280., 380., 480., 780., 980.],
            'LOWER_BOUND/1': [20., 20., 20., 20., 20., 20., 20., 20.],
            'RAIN_SHAPE/1': [1., 1., 1., 1., 1., 1., 1., 1.],
            'RAIN_SCALE/1': [ 5.,  3.,  9.,  7., 15., 13., 25., 30.],
            'DOWNSTREAM/2': [
                [False, False, False, False, False, True, False, False],
                [False, False, True, False, False, False, False, False],
                [False, False, False, False, True, False, False, False],
                [False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, True, False],
                [False, False, False, False, False, False, True, False],
                [False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, False, False]
            ],
            'SINK_RES/1': [False, False, False, False, False, False, False, True],
            'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT/0': 0.05,
            'LOW_PENALTY/1': [-5., -5., -5., -5., -5., -5., -5., -5.],
            'HIGH_PENALTY/1': [-10., -10., -10., -10., -10., -10., -10., -10.]
        }
        with tf.Session(graph=self.graph1) as sess:
            for name, tensor in nf.items():
                value = sess.run(tensor)
                list1 = list(value.flatten())
                list2 = list(np.array(expected_initializers[name]).flatten())
                for v1, v2 in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2)

    def test_instantiate_initial_state_fluents(self):
        self.compiler1._build_object_table()
        sf = self.compiler1._instantiate_initial_state_fluents()

        expected_state_fluents = {
            'rlevel/1': { 'shape': (8,) , 'dtype': tf.float32 }
        }
        self.assertIsInstance(sf, dict)
        self.assertEqual(len(sf), len(expected_state_fluents))
        for name, tensor in sf.items():
            self.assertIn(name, expected_state_fluents)
            shape = expected_state_fluents[name]['shape']
            dtype = expected_state_fluents[name]['dtype']
            self.assertEqual(tensor.name, '{}:0'.format(name))
            self.assertIsInstance(tensor, tf.Tensor)
            self.assertEqual(tensor.dtype, dtype)
            self.assertEqual(tensor.shape, shape)

        expected_initializers = {
            'rlevel/1': [75., 50., 50., 50., 50., 50., 50., 50.]
        }
        with tf.Session(graph=self.graph1) as sess:
            for name, tensor in sf.items():
                value = sess.run(tensor)
                list1 = list(value.flatten())
                list2 = list(np.array(expected_initializers[name]).flatten())
                for v1, v2 in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2)

    def test_instantiate_default_action_fluents(self):
        self.compiler1._build_object_table()
        af = self.compiler1._instantiate_default_action_fluents()

        expected_action_fluents = {
            'outflow/1': { 'shape': (8,) , 'dtype': tf.float32 }
        }
        self.assertIsInstance(af, dict)
        self.assertEqual(len(af), len(expected_action_fluents))
        for name, tensor in af.items():
            self.assertIn(name, expected_action_fluents)
            shape = expected_action_fluents[name]['shape']
            dtype = expected_action_fluents[name]['dtype']
            self.assertEqual(tensor.name, '{}:0'.format(name))
            self.assertIsInstance(tensor, tf.Tensor)
            self.assertEqual(tensor.dtype, dtype)
            self.assertEqual(tensor.shape, shape)

        expected_initializers = {
            'outflow/1': [0., 0., 0., 0., 0., 0., 0., 0.]
        }
        with tf.Session(graph=self.graph1) as sess:
            for name, tensor in af.items():
                value = sess.run(tensor)
                list1 = list(value.flatten())
                list2 = list(np.array(expected_initializers[name]).flatten())
                for v1, v2 in zip(list1, list2):
                    self.assertAlmostEqual(v1, v2)

    def test_compile_expressions(self):
        expected = {
            # rddl1: RESERVOIR ====================================================
            'rainfall/1':   { 'shape': [8], 'dtype': tf.float32, 'scope': ['?r'] },
            'evaporated/1': { 'shape': [8], 'dtype': tf.float32, 'scope': ['?r'] },
            'overflow/1':   { 'shape': [8], 'dtype': tf.float32, 'scope': ['?r'] },
            "rlevel'/1":    { 'shape': [8], 'dtype': tf.float32, 'scope': ['?r'] },

            # rddl2: MARS ROVER ===================================================
            "xPos'/0":   { 'shape': [], 'dtype': tf.float32, 'scope': [] },
            "yPos'/0":   { 'shape': [], 'dtype': tf.float32, 'scope': [] },
            "time'/0":   { 'shape': [], 'dtype': tf.float32, 'scope': [] },
            "picTaken'/1": { 'shape': [3], 'dtype': tf.bool, 'scope': ['?p'] }
        }

        compilers = [self.compiler1, self.compiler2]
        rddls = [self.rddl1, self.rddl2]
        for compiler, rddl in zip(compilers, rddls):
            compiler._build_object_table()

            nf = compiler._instantiate_non_fluents()
            sf = compiler._instantiate_initial_state_fluents()
            af = compiler._instantiate_default_action_fluents()
            scope = {}
            scope.update(nf)
            scope.update(sf)
            scope.update(af)

            _, cpfs = rddl.domain.cpfs
            for cpf in cpfs:
                name = cpf.name
                expr = cpf.expr
                t = compiler._compile_expression(expr, scope)
                scope[name] = t.tensor
                self.assertIsInstance(t, TensorFluent)
                self.assertEqual(t.shape, expected[name]['shape'])
                self.assertEqual(t.dtype, expected[name]['dtype'])
                self.assertEqual(t.scope.as_list(), expected[name]['scope'])

            reward_expr = rddl.domain.reward
            t = compiler._compile_expression(reward_expr, scope)
            self.assertIsInstance(t, TensorFluent)
            self.assertEqual(t.shape, [])
            self.assertEqual(t.dtype, tf.float32)
            self.assertEqual(t.scope.as_list(), [])
