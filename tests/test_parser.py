from tfrddlsim import parser
from tfrddlsim.rddl import RDDL, Domain, Instance, NonFluents

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


class TestRDDLyacc(unittest.TestCase):

    def setUp(self):
        rddl = '''
        ////////////////////////////////////////////////////////////////////
        // The problem models the active maintenance of water levels in
        // a Reservoir system with uncertain rainfall and nonlinear
        // evaporation rates as a function of water level.  The objective
        // is to maintain all reservoir levels within a desired safe range.
        //
        // The state of each reservoir is the water level (rlevel).  The
        // actions are to set the outflows of each reservoir.  Rewards
        // are summed per reservoir and optimal when the water level is
        // within predefined upper and lower bounds.
        //
        // Author: Ga Wu, Buser Say inspired by Aswin Raghavan's RDDL model
        ////////////////////////////////////////////////////////////////////

        domain reservoir {
            requirements = {
                concurrent,           // x and y directions move independently and simultaneously
                reward-deterministic, // this domain does not use a stochastic reward
                intermediate-nodes,   // this domain uses intermediate pvariable nodes
                constrained-state     // this domain uses state constraints
            };
        }

        non-fluents res8 { }

        instance inst_reservoir_res8 { }
        '''
        self.parser = parser.RDDLParser()
        self.parser.build()
        self.rddl = self.parser.parse(rddl)

    def test_rddl(self):
        self.assertIsInstance(self.rddl, RDDL)
        self.assertIsInstance(self.rddl.domain, Domain)
        self.assertIsInstance(self.rddl.instance, Instance)
        self.assertIsInstance(self.rddl.non_fluents, NonFluents)

    def test_domain_block(self):
        domain = self.rddl.domain
        self.assertIsInstance(domain, Domain)
        self.assertEqual(domain.name, 'reservoir')

    def test_requirements_section(self):
        requirements = self.rddl.domain.requirements
        self.assertListEqual(sorted(requirements), sorted(['concurrent', 'reward-deterministic', 'intermediate-nodes', 'constrained-state']))

    def test_instance_block(self):
        instance = self.rddl.instance
        self.assertIsInstance(instance, Instance)
        self.assertEqual(instance.name, 'inst_reservoir_res8')

    def test_nonfluents_block(self):
        non_fluents = self.rddl.non_fluents
        self.assertIsInstance(non_fluents, NonFluents)
        self.assertEqual(non_fluents.name, 'res8')
