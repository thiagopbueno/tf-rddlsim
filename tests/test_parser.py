from tfrddlsim import parser
from tfrddlsim.rddl import RDDL, Domain, Instance, NonFluents
from tfrddlsim.pvariable import NonFluent

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

            types {
                res: object;
                picture-point : object;
                x_pos : object;
                y_pos : object;
                crowdlevel : {@low, @med, @high};
                enum_level : {@low, @high}; // An enumerated type
            };

            pvariables {

                // Constants
                MAX_RES_CAP(res): { non-fluent, real, default = 100.0 }; // Beyond this amount, water spills over
                UPPER_BOUND(res): { non-fluent, real, default = 80.0 };  // The upper bound for a safe reservoir level
                LOWER_BOUND(res): { non-fluent, real, default = 20.0 };  // The lower bound for a safe reservoir level
                RAIN_SHAPE(res):  { non-fluent, real, default = 25.0 };  // Gamma shape parameter for rainfall
                RAIN_SCALE(res):  { non-fluent, real, default = 25.0 };  // Gamma scale paramater for rainfall
                DOWNSTREAM(res,res): { non-fluent, bool, default = false }; // Indicates 2nd res is downstream of 1st res
                SINK_RES(res):    { non-fluent, bool, default = false }; // This is a "sink" water source (sea, ocean) 
                MAX_WATER_EVAP_FRAC_PER_TIME_UNIT: { non-fluent, real, default = 0.05 }; // Maximum fraction of evaporation
                LOW_PENALTY(res) : { non-fluent, real, default =  -5.0 }; // Penalty per unit of level < LOWER_BOUND
                HIGH_PENALTY(res): { non-fluent, real, default = -10.0 }; // Penalty per unit of level > UPPER_BOUND

                // Each picture occurs in a different place and awards a different value
                PICT_XPOS(picture-point)   : { non-fluent, real, default = 0.0 };
                PICT_YPOS(picture-point)   : { non-fluent, real, default = 0.0 };

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
        self.assertListEqual(requirements, ['concurrent', 'reward-deterministic', 'intermediate-nodes', 'constrained-state'])

    def test_types_section(self):
        types = self.rddl.domain.types
        expected = [
            ('res', 'object'),
            ('picture-point', 'object'),
            ('x_pos', 'object'),
            ('y_pos', 'object'),
            ('crowdlevel', ['@low', '@med', '@high']),
            ('enum_level', ['@low', '@high'])
        ]
        for t in expected:
            self.assertIn(t, types)

    def test_pvariables_section(self):
        pvariables = self.rddl.domain.pvariables

        expected = {
            'MAX_RES_CAP': { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' : 100.0 },
            'UPPER_BOUND': { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' : 80.0 },
            'LOWER_BOUND': { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' : 20.0 },
            'RAIN_SHAPE':  { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' : 25.0 },
            'RAIN_SCALE':  { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' : 25.0 },
            'DOWNSTREAM':  { 'params': ['res', 'res'], 'type': 'non-fluent', 'range': 'bool', 'default' : False },
            'SINK_RES':    { 'params': ['res'], 'type': 'non-fluent', 'range': 'bool', 'default' : False },
            'MAX_WATER_EVAP_FRAC_PER_TIME_UNIT': { 'params': [], 'type': 'non-fluent', 'range': 'real', 'default' : 0.05 },
            'LOW_PENALTY' : { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' :  -5.0 },
            'HIGH_PENALTY': { 'params': ['res'], 'type': 'non-fluent', 'range': 'real', 'default' : -10.0 },
            'PICT_XPOS'   : { 'params': ['picture-point'], 'type': 'non-fluent', 'range': 'real', 'default': 0.0 },
            'PICT_YPOS'   : { 'params': ['picture-point'], 'type': 'non-fluent', 'range': 'real', 'default': 0.0 }
        }

        for pvar in pvariables:
            if pvar.param_types is None:
                self.assertEqual(pvar.arity(), 0)
            else:
                self.assertEqual(pvar.arity(), len(pvar.param_types))
            if pvar.range == 'bool':
                self.assertIsInstance(pvar.def_value, bool)
            elif pvar.range == 'real':
                self.assertIsInstance(pvar.def_value, float)
            elif pvar.range == 'int':
                self.assertIsInstance(pvar.def_value, int)

            pvar_params = expected[pvar.name]['params']
            pvar_type = expected[pvar.name]['type']
            pvar_range = expected[pvar.name]['range']
            pvar_def_value = expected[pvar.name]['default']
            self.assertIn(pvar.name, expected)
            if len(pvar_params) == 0:
                self.assertIsNone(pvar.param_types)
            else:
                self.assertListEqual(pvar.param_types, pvar_params)
            if pvar_type == 'non-fluent':
                self.assertIsInstance(pvar, NonFluent)
            self.assertEqual(pvar.range, pvar_range)
            self.assertAlmostEqual(pvar.def_value, pvar_def_value)

    def test_instance_block(self):
        instance = self.rddl.instance
        self.assertIsInstance(instance, Instance)
        self.assertEqual(instance.name, 'inst_reservoir_res8')

    def test_nonfluents_block(self):
        non_fluents = self.rddl.non_fluents
        self.assertIsInstance(non_fluents, NonFluents)
        self.assertEqual(non_fluents.name, 'res8')
