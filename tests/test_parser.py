from tfrddlsim import parser

import unittest

class TestRDDLlex(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open('rddl/Reservoir.rddl', mode='r') as file:
            cls.reservoir_data = file.read()

        with open('rddl/Mars_Rover.rddl', mode='r') as file:
            cls.mars_rover_data = file.read()

    def setUp(self):
        self.lexer = parser.RDDLlex()
        self.lexer.build()

    def test_newlines(self):
        self.lexer.input(self.reservoir_data)
        for _ in self.lexer(): pass
        self.assertEqual(self.lexer._lexer.lineno, 145)
