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
