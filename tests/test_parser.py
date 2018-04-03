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

    def test_identifiers(self):
        self.lexer.input(self.reservoir_data)
        for tok in self.lexer():
            if tok.type == 'ID':
                self.assertIsInstance(tok.value, str)
                self.assertIn(tok.value[0], "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ")
                if len(tok.value) > 1:
                    for c in tok.value[1:-1]:
                        self.assertIn(c, "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789-_")
                    self.assertIn(tok.value[-1], "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789'")

    def test_reserved_words(self):
        self.lexer.input(self.reservoir_data)
        for tok in self.lexer():
            if tok.type == 'ID':
                self.assertNotIn(tok.value, self.lexer.reserved)
            elif tok.value in self.lexer.reserved:
                self.assertEqual(tok.type, self.lexer.reserved[tok.value])

    def test_integer_numbers(self):
        self.lexer.input(self.reservoir_data)
        for tok in self.lexer():
            if tok.type == 'INTEGER':
                self.assertIsInstance(tok.value, int)
            elif isinstance(tok.value, int):
                self.assertEqual(tok.type, 'INTEGER')
