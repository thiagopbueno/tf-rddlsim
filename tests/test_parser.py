from tfrddlsim import parser

import unittest


with open('rddl/Reservoir.rddl', mode='r') as file:
    RESERVOIR = file.read()

with open('rddl/Mars_Rover.rddl', mode='r') as file:
    MARS_ROVER = file.read()


class TestRDDLlex(unittest.TestCase):

    def setUp(self):
        self.lexer = parser.RDDLlex()
        self.lexer.build()

    def test_newlines(self):
        self.lexer.input(RESERVOIR)
        for _ in self.lexer(): pass
        self.assertEqual(self.lexer._lexer.lineno, 145)

    def test_identifiers(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if tok.type == 'ID':
                self.assertIsInstance(tok.value, str)
                self.assertIn(tok.value[0], "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ")
                if len(tok.value) > 1:
                    for c in tok.value[1:-1]:
                        self.assertIn(c, "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789-_")
                    self.assertIn(tok.value[-1], "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789'")

    def test_reserved_words(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if tok.type == 'ID':
                self.assertNotIn(tok.value, self.lexer.reserved)
            if tok.value in self.lexer.reserved:
                self.assertEqual(tok.type, self.lexer.reserved[tok.value])

    def test_floating_point_numbers(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if tok.type == 'DOUBLE':
                self.assertIsInstance(tok.value, float)
            if isinstance(tok.value, float):
                self.assertEqual(tok.type, 'DOUBLE')

    def test_integer_numbers(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if tok.type == 'INTEGER':
                self.assertIsInstance(tok.value, int)
            if isinstance(tok.value, int):
                self.assertEqual(tok.type, 'INTEGER')

    def test_operators(self):
        op2tok = {
            '^': 'AND',
            '|': 'OR',
            '~': 'NOT',
            '+': 'PLUS',
            '*': 'TIMES',
            '.': 'DOT',
            '=>': 'IMPLY',
            '<=>': 'EQUIV',
            '~=': 'NEQ',
            '<=': 'LESSEQ',
            '<': 'LESS',
            '>=': 'GREATEREQ',
            '>': 'GREATER',
            '=': 'ASSIGN_EQUAL',
            '==': 'COMP_EQUAL',
            '/': 'DIV',
            '-': 'MINUS',
            ':': 'COLON',
            ';': 'SEMI',
            '$': 'DOLLAR_SIGN',
            '?': 'QUESTION',
            '&': 'AMPERSAND'
        }

        tok2op = {
            'AND': '^',
            'OR': '|',
            'NOT': '~',
            'PLUS': '+',
            'TIMES': '*',
            'DOT': '.',
            'IMPLY': '=>',
            'EQUIV': '<=>',
            'NEQ': '~=',
            'LESSEQ': '<=',
            'LESS': '<',
            'GREATEREQ': '>=',
            'GREATER': '>',
            'ASSIGN_EQUAL': '=',
            'COMP_EQUAL': '==',
            'DIV': '/',
            'MINUS': '-',
            'COLON': ':',
            'SEMI': ';',
            'DOLLAR_SIGN': '$',
            'QUESTION': '?',
            'AMPERSAND': '&'
        }

        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if tok.value in op2tok:
                self.assertEqual(tok.type, op2tok[tok.value])
            if tok.type in tok2op:
                self.assertEqual(tok.value, tok2op[tok.type])

    def test_delimiters(self):
        delim2tok = {
            '(': 'LPAREN',
            ')': 'RPAREN',
            '{': 'LCURLY',
            '}': 'RCURLY',
            ',': 'COMMA',
            '[': 'LBRACK',
            ']': 'RBRACK'
        }

        tok2delim = {
            'LPAREN': '(',
            'RPAREN': ')',
            'LCURLY': '{',
            'RCURLY': '}',
            'COMMA': ',',
            'LBRACK': '[',
            'RBRACK': ']'
        }

        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if tok.value in delim2tok:
                self.assertEqual(tok.type, delim2tok[tok.value])
            if tok.type in tok2delim:
                self.assertEqual(tok.value, tok2delim[tok.type])

    def test_variables(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if not isinstance(tok.value, str):
                continue

            if tok.type == 'VAR':
                self.assertIsInstance(tok.value, str)
                self.assertGreaterEqual(len(tok.value), 2)
                self.assertEqual(tok.value[0], '?')
                if len(tok.value) == 2:
                    self.assertIn(tok.value[1], "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789")
                else:
                    for c in tok.value[1:-1]:
                        self.assertIn(c, "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789-_")
                    self.assertIn(tok.value[-1], "abcdefghijklmnopqrstuvxwyzABCDEFGHIJKLMNOPQRSTUVXWYZ0123456789")

            if tok.value[0] == '?':
                self.assertEqual(tok.type, 'VAR')

    def test_ignore_whitespaces(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if isinstance(tok.value, str):
                self.assertNotIn(' ', tok.value)
                self.assertNotIn('\t', tok.value)

    def test_ignore_comments(self):
        self.lexer.input(RESERVOIR)
        for tok in self.lexer():
            if isinstance(tok.value, str):
                self.assertFalse(tok.value.startswith("//"))
