from tfrddlsim.parser import RDDLParser
from tfrddlsim.compiler import Compiler

import tensorflow as tf

import unittest


class TestCompiler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('rddl/Reservoir.rddl', mode='r') as file:
            RESERVOIR = file.read()

        parser = RDDLParser()
        parser.build()
        cls.rddl = parser.parse(RESERVOIR)

    def setUp(self):
        self.graph = tf.Graph()
        self.compiler = Compiler(self.rddl, self.graph)
        self.assertIs(self.compiler._rddl, self.rddl)
        self.assertIs(self.compiler._graph, self.graph)

    def test_build_object_table(self):
        self.compiler._build_object_table()
        self.assertIn('res', self.compiler._object_table)
        size = self.compiler._object_table['res']['size']
        idx = self.compiler._object_table['res']['idx']
        self.assertEqual(size, 8)
        objs = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8']
        for i, obj in enumerate(objs):
            self.assertIn(obj, idx)
            self.assertEqual(idx[obj], i)
