from tfrddlsim.rddl import RDDL
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
        self.assertIn('res', self.compiler1.object_table)
        size = self.compiler1.object_table['res']['size']
        idx = self.compiler1.object_table['res']['idx']
        self.assertEqual(size, 8)
        objs = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
        for i, obj in enumerate(objs):
            self.assertIn(obj, idx)
            self.assertEqual(idx[obj], i)

    def test_build_action_preconditions_table(self):
        local_preconds = self.compiler1.local_action_preconditions
        self.assertIsInstance(local_preconds, dict)
        self.assertEqual(len(local_preconds), 1)
        self.assertIn('outflow/1', local_preconds)
        self.assertEqual(len(local_preconds['outflow/1']), 2)

        global_preconds = self.compiler1.global_action_preconditions
        self.assertIsInstance(global_preconds, list)
        self.assertEqual(len(global_preconds), 0)

    def test_instantiate_non_fluents(self):
        nf = self.compiler1.non_fluents

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
        sf = dict(self.compiler1.initial_state_fluents)

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
        action_fluents = self.compiler1.default_action_fluents
        self.assertIsInstance(action_fluents, list)
        for fluent in action_fluents:
            self.assertIsInstance(fluent, tuple)
            self.assertEqual(len(fluent), 2)
            self.assertIsInstance(fluent[0], str)
            self.assertIsInstance(fluent[1], tf.Tensor)

        af = dict(action_fluents)
        print(af)

        expected_action_fluents = {
            'outflow/1': { 'shape': (8,) , 'dtype': tf.float32 }
        }
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

    def test_state_fluent_ordering(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            initial_state_fluents = dict(compiler.initial_state_fluents)
            current_state_ordering = compiler.state_fluent_ordering
            self.assertEqual(len(current_state_ordering), len(initial_state_fluents))
            for fluent in initial_state_fluents:
                self.assertIn(fluent, current_state_ordering)

            next_state_ordering = compiler.next_state_fluent_ordering
            self.assertEqual(len(current_state_ordering), len(next_state_ordering))

            for current_fluent, next_fluent in zip(current_state_ordering, next_state_ordering):
                self.assertEqual(RDDL.rename_state_fluent(current_fluent), next_fluent)
                self.assertEqual(RDDL.rename_next_state_fluent(next_fluent), current_fluent)

    def test_action_fluent_ordering(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            default_action_fluents = dict(compiler.default_action_fluents)
            action_fluent_ordering = compiler.action_fluent_ordering
            self.assertEqual(len(action_fluent_ordering), len(default_action_fluents))
            for action_fluent in action_fluent_ordering:
                self.assertIn(action_fluent, default_action_fluents)

    def test_state_size(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            state_size = compiler.state_size
            initial_state_fluents = dict(compiler.initial_state_fluents)
            state_fluent_ordering = compiler.state_fluent_ordering
            next_state_fluent_ordering = compiler.next_state_fluent_ordering

            self.assertIsInstance(state_size, tuple)
            for shape in state_size:
                self.assertIsInstance(shape, tf.TensorShape)
            self.assertEqual(len(state_size), len(initial_state_fluents))
            self.assertEqual(len(state_size), len(state_fluent_ordering))
            self.assertEqual(len(state_size), len(next_state_fluent_ordering))

            for shape, name in zip(state_size, state_fluent_ordering):
                actual = shape.as_list()
                expected = initial_state_fluents[name].shape.as_list()
                self.assertListEqual(actual, expected)

            scope = {}
            scope.update(compiler.non_fluents)
            scope.update(dict(compiler.initial_state_fluents))
            scope.update(dict(compiler.default_action_fluents))
            next_state_fluents = dict(compiler.compile_cpfs(scope))
            for shape, name in zip(state_size, next_state_fluent_ordering):
                actual = shape.as_list()
                expected = next_state_fluents[name].shape.as_list()
                self.assertListEqual(actual, expected)

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
            nf = compiler.non_fluents
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
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

    def test_compile_cpfs(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            nf = compiler.non_fluents
            sf = dict(compiler.initial_state_fluents)
            af = dict(compiler.default_action_fluents)
            scope = {}
            scope.update(nf)
            scope.update(sf)
            scope.update(af)

            next_state_fluents = compiler.compile_cpfs(scope)
            self.assertIsInstance(next_state_fluents, list)
            for cpf in next_state_fluents:
                self.assertIsInstance(cpf, tuple)
            self.assertEqual(len(next_state_fluents), len(sf))

            next_state_fluents = dict(next_state_fluents)
            for fluent in sf:
                next_fluent = RDDL.rename_state_fluent(fluent)
                self.assertIn(next_fluent, next_state_fluents)
                self.assertIsInstance(next_state_fluents[next_fluent], tf.Tensor)

    def test_compile_reward(self):
        compilers = [self.compiler1, self.compiler2]
        for compiler in compilers:
            scope = {}
            scope.update(compiler.non_fluents)
            scope.update(dict(compiler.initial_state_fluents))
            scope.update(dict(compiler.default_action_fluents))
            next_state_fluents = dict(compiler.compile_cpfs(scope))
            scope.update(next_state_fluents)
            reward = compiler.compile_reward(scope)
            self.assertIsInstance(reward, tf.Tensor)
            self.assertEqual(reward.shape.as_list(), [])
