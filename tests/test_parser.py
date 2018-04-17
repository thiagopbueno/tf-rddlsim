from tfrddlsim import parser
from tfrddlsim.rddl import RDDL, Domain, Instance, NonFluents
from tfrddlsim.pvariable import NonFluent, StateFluent, ActionFluent, IntermediateFluent
from tfrddlsim.cpf import CPF

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
                PICT_XPOS(picture-point)   : { non-fluent, real, default = 0.0 };
                PICT_YPOS(picture-point)   : { non-fluent, real, default = 0.0 };

                // State fluents
                rlevel(res): {state-fluent, real, default = 50.0 }; // Reservoir level for res
                xPos : { state-fluent, real, default = 0.0 };
                yPos : { state-fluent, real, default = 0.0 };
                time : { state-fluent, real, default = 0.0 };
                picTaken(picture-point) : { state-fluent, bool, default = false };

                // Action fluents
                outflow(res): { action-fluent, real, default = 0.0 }; // Action to set outflow of res
                xMove       : { action-fluent, real, default = 0.0 };
                yMove       : { action-fluent, real, default = 0.0 };
                snapPicture : { action-fluent, bool, default = false };

                // Intermediate fluents
                evaporated(res): {interm-fluent, real, level=1}; // How much evaporates from res in this time step?
                rainfall(res):   {interm-fluent, real, level=1}; // How much rainfall is there in this time step?
                overflow(res):   {interm-fluent, real, level=1}; // Is there any excess overflow (over the rim)?
            };

            cpfs {
                evaporated(?r) = MAX_WATER_EVAP_FRAC_PER_TIME_UNIT
                                 *[(-11.8 * rlevel(?r)*rlevel(?r))/(MAX_RES_CAP(?r)*MAX_RES_CAP(?r) - 5)]
                                 * (+ rlevel(?r));

                // Consider MAX_RES_CAP=90, rlevel=100, outflow=4, then the excess overflow is 6 units
                // Consider MAX_RES_CAP=100, rlevel=90, outflow=4, then the excess overflow is 0 units
                overflow(?r) = max[0, rlevel(?r) - outflow(?x) - MAX_RES_CAP(?r, ?t)];

                rlevel'(?r) = rlevel(?r) + rainfall(?r) + (- evaporated(?r)) - outflow(?r) + [- overflow(?r)];

                distance(?r) = sqrt[pow[(location(?l)-CENTER(?l)),2]];
                scalefactor = 2.0/(1.0+exp[-2*distance])-0.99;

                rainfall(?r, ?s) = Gamma(RAIN_SHAPE(?r, ?s) - (- 2), 0.1 * RAIN_SCALE(?s));

                xPos' = xPos + xMove + Normal(0.0, MOVE_VARIANCE_MULT*abs[xMove]);
                yPos' = cos[yPos + exp[yMove + (-Normal(1.0, abs[yMove] - (10 * MOVE_VARIANCE_MULT)))]];

                // Choose a level with following probabilities
                i2 = Discrete(enum_level,
                                @low : 0.5 + i1,
                                @high : 0.3,
                                @medium : - i1 + 0.2
                            );

                i1 = KronDelta(p + Bernoulli( (p + q + r)/3.0 ) + r);  // Just set i1 to a count of true state variables

                picTaken'(?p) = picTaken(?p) == true | ~notPicTaken(?p) &
                        [~snapPicture ~= false ^ (time <= MAX_TIME)
                         & (PICT_ERROR_ALLOW(?p) > abs[xPos - PICT_XPOS(?p)])
                         ^ ~(abs[yPos - PICT_YPOS(?p)] == PICT_ERROR_ALLOW(?p))];

                time' = if (snapPicture)
                    then (time + 0.25)
                    else (time + abs[xMove] + abs[yMove]);

                j2 = Discrete(enum_level,
                        @high : 0.3,
                        @low : if (i1 >= 2) then 0.5 else 0.2
                    );

                // Conditional linear stochastic equation
                o2 = switch (i2) {
                    case @high   : i1 + 3.0 + Normal(0.0, i1*i1/4.0),
                    case @medium : -i2 + 2 + Normal(1.0, i2*i1/2.0)
                };

                o3 = switch (i2) {
                    case @high   : i1 + 3.0 + Normal(0.0, i1*i1/4.0),
                    case @medium : -i2 + 2 + Normal(1.0, i2*i1/2.0),
                    default : -Normal(1.0, 0.0) * (-16.0)
                };

                rlevel2'(?r) = sum_{?up : res} [DOWNSTREAM(?up,?r) * (outflow(?up) + overflow(?up))];

                rlevel3'(?r) = rlevel(?r) + rainfall(?r) - evaporated(?r) - outflow(?r) - overflow(?r)
                      + sum_{?up : res} [DOWNSTREAM(?up,?r)*(outflow(?up) + overflow(?up))];

                rlevel4'(?r) = rlevel(?r) + rainfall(?r) - evaporated(?r) - outflow(?r)
                      + sum_{?up : res} [DOWNSTREAM(?up,?r)*(outflow(?up) + overflow(?up))]
                      - overflow(?r);

                rlevel5'(?r) = rlevel(?r) + rainfall(?r) - evaporated(?r) - outflow(?r)
                      + (sum_{?up : res} [DOWNSTREAM(?up,?r)*(outflow(?up) + overflow(?up))])
                      - overflow(?r);

                rlevel6'(?r) = max_{?up : res, ?down : res2} [DOWNSTREAM(?up,?down) * outflow(?up) + overflow(?up)];

                rlevel7'(?r) = rlevel(?r) + rainfall(?r) - evaporated(?r) - outflow(?r)
                      + sum_{?up : res1, ?down : res} [DOWNSTREAM(?up,?down)*(outflow(?up) + overflow(?up))]
                      - overflow(?r);


                // skill_teaching_mdp.rddl
                hintedRight'(?s) =
                    KronDelta( [forall_{?s3: skill} ~updateTurn(?s3)] ^
                                giveHint(?s) ^
                                forall_{?s2: skill}[PRE_REQ(?s2, ?s) => proficiencyHigh(?s2)] );

                hintDelayVar'(?s) =
                    KronDelta( [forall_{?s2: skill} ~updateTurn(?s2)] ^ giveHint(?s) );

                // crossing_traffic_mdp.rddl
                robot-at'(?x,?y) =
                    if ( exists_{?x2 : xpos, ?y2 : ypos} [ GOAL(?x2,?y2) ^ robot-at(?x2,?y2)  ] )
                    then
                        KronDelta(false) // because of fall-through we know (?x,y) != (?x2,?y2)
                    // Check for legal robot movement (robot disappears if at an obstacle)
                    else if ( move-north ^ exists_{?y2 : ypos} [ NORTH(?y2,?y) ^ robot-at(?x,?y2) ^ ~obstacle-at(?x,?y2) ] )
                    then
                        KronDelta(true) // robot moves to this location
                    else
                        false;

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
            'PICT_YPOS'   : { 'params': ['picture-point'], 'type': 'non-fluent', 'range': 'real', 'default': 0.0 },
            'rlevel': { 'params': ['res'], 'type': 'state-fluent', 'range': 'real', 'default': 50.0 },
            'xPos' : { 'params': [], 'type': 'state-fluent', 'range': 'real', 'default': 0.0 },
            'yPos' : { 'params': [], 'type': 'state-fluent', 'range': 'real', 'default': 0.0 },
            'time' : { 'params': [], 'type': 'state-fluent', 'range': 'real', 'default': 0.0 },
            'picTaken' : { 'params': ['picture-point'], 'type': 'state-fluent', 'range': 'bool', 'default': False },
            'outflow'     : { 'params': ['res'], 'type': 'action-fluent', 'range': 'real', 'default': 0.0 },
            'xMove'       : { 'params': [], 'type': 'action-fluent', 'range': 'real', 'default': 0.0 },
            'yMove'       : { 'params': [], 'type': 'action-fluent', 'range': 'real', 'default': 0.0 },
            'snapPicture' : { 'params': [], 'type': 'action-fluent', 'range': 'bool', 'default': False },
            'evaporated': { 'params': ['res'], 'type': 'interm-fluent', 'range': 'real', 'level': 1},
            'rainfall':   { 'params': ['res'], 'type': 'interm-fluent', 'range': 'real', 'level': 1},
            'overflow':   { 'params': ['res'], 'type': 'interm-fluent', 'range': 'real', 'level': 1}
        }

        for pvar in pvariables:
            pvar_params = expected[pvar.name]['params']
            pvar_type = expected[pvar.name]['type']
            pvar_range = expected[pvar.name]['range']
            pvar_def_value = expected[pvar.name]['default'] if pvar_type != 'interm-fluent' else None
            pvar_level = expected[pvar.name]['level'] if pvar_type == 'interm-fluent' else None

            # name
            self.assertIn(pvar.name, expected)

            # params
            if len(pvar_params) == 0:
                self.assertIsNone(pvar.param_types)
            else:
                self.assertListEqual(pvar.param_types, pvar_params)

            # type
            if pvar_type == 'non-fluent':
                self.assertIsInstance(pvar, NonFluent)
            elif pvar_type == 'state-fluent':
                self.assertIsInstance(pvar, StateFluent)
            elif pvar_type == 'action-fluent':
                self.assertIsInstance(pvar, ActionFluent)
            elif pvar_type == 'interm-fluent':
                self.assertIsInstance(pvar, IntermediateFluent)

            # range
            self.assertEqual(pvar.range, pvar_range)

            # default value
            if pvar_type != 'interm-fluent':
                self.assertAlmostEqual(pvar.def_value, pvar_def_value)
                if pvar.range == 'bool':
                    self.assertIsInstance(pvar.def_value, bool)
                elif pvar.range == 'real':
                    self.assertIsInstance(pvar.def_value, float)
                elif pvar.range == 'int':
                    self.assertIsInstance(pvar.def_value, int)

            # level
            if pvar_type == 'interm-fluent':
                self.assertEqual(pvar.level, pvar_level)
                self.assertIsInstance(pvar.level, int)

    def test_cpfs_section(self):
        header, cpfs = self.rddl.domain.cpfs
        self.assertEqual(header, 'cpfs')
        for cpf in cpfs:
            self.assertEqual(cpf.pvar[0], 'pvar_expr')

        ast = {
            'evaporated': [
                '*',
                '*',
                ('MAX_WATER_EVAP_FRAC_PER_TIME_UNIT', None),
                '/',
                '*',
                '*',
                '-',
                11.8,
                ('rlevel', ['?r']),
                ('rlevel', ['?r']),
                '-',
                '*',
                ('MAX_RES_CAP', ['?r']),
                ('MAX_RES_CAP', ['?r']),
                5,
                '+',
                ('rlevel', ['?r'])
            ],
            "rlevel'": [
                '+',
                '-',
                '+',
                '+',
                ('rlevel', ['?r']),
                ('rainfall', ['?r']),
                '-',
                ('evaporated', ['?r']),
                ('outflow', ['?r']),
                '-',
                ('overflow', ['?r'])
            ],
            'overflow': [
                'max',
                0,
                '-',
                '-',
                ('rlevel', ['?r']),
                ('outflow', ['?x']),
                ('MAX_RES_CAP', ['?r', '?t'])
            ],
            'distance': [
                'sqrt',
                'pow',
                '-',
                ('location', ['?l']),
                ('CENTER', ['?l']),
                2
            ],
            'scalefactor': [
                '-',
                '/',
                2.0,
                '+',
                1.0,
                'exp',
                '*',
                '-',
                2,
                ('distance', None),
                0.99
            ],
            'rainfall': [
                'Gamma',
                '-',
                ('RAIN_SHAPE', ['?r', '?s']),
                '-',
                2,
                '*',
                0.1,
                ('RAIN_SCALE', ['?s'])
            ],
            "xPos'": [
                '+',
                '+',
                ('xPos', None),
                ('xMove', None),
                'Normal',
                0.0,
                '*',
                ('MOVE_VARIANCE_MULT', None),
                'abs',
                ('xMove', None)
            ],
            "yPos'": [
                'cos',
                '+',
                ('yPos', None),
                'exp',
                '+',
                ('yMove', None),
                '-',
                'Normal',
                1.0,
                '-',
                'abs',
                ('yMove', None),
                '*',
                10,
                ('MOVE_VARIANCE_MULT', None)
            ],
            'i1': [
                'KronDelta',
                '+',
                '+',
                ('p', None),
                'Bernoulli',
                '/',
                '+',
                '+',
                ('p', None),
                ('q', None),
                ('r', None),
                3.0,
                ('r', None)
            ],
            'i2': [
                'Discrete',
                'enum_level',
                '@low',
                '+',
                0.5,
                ('i1', None),
                '@high',
                0.3,
                '@medium',
                '+',
                '-',
                ('i1', None),
                0.2
            ],
            "picTaken'": [
                '|',
                '==',
                ('picTaken', ['?p']),
                True,
                '&',
                '~',
                ('notPicTaken', ['?p']),
                '^',
                '&',
                '^',
                '~=',
                '~',
                ('snapPicture', None),
                False,
                '<=',
                ('time', None),
                ('MAX_TIME', None),
                '>',
                ('PICT_ERROR_ALLOW', ['?p']),
                'abs',
                '-',
                ('xPos', None),
                ('PICT_XPOS', ['?p']),
                '~',
                '==',
                'abs',
                '-',
                ('yPos', None),
                ('PICT_YPOS', ['?p']),
                ('PICT_ERROR_ALLOW', ['?p'])
            ],
            "time'": [
                'if',
                ('snapPicture', None),
                '+',
                ('time', None),
                0.25,
                '+',
                '+',
                ('time', None),
                'abs',
                ('xMove', None),
                'abs',
                ('yMove', None)
            ],
            'j2': [
                'Discrete',
                'enum_level',
                '@high',
                0.3,
                '@low',
                'if',
                '>=',
                ('i1', None),
                2,
                0.5,
                0.2
            ],
            'o2': [
                'switch',
                ('i2', None),
                '@high',
                '+',
                '+',
                ('i1', None),
                3.0,
                'Normal',
                0.0,
                '/',
                '*',
                ('i1', None),
                ('i1', None),
                4.0,
                '@medium',
                '+',
                '+',
                '-',
                ('i2', None),
                2,
                'Normal',
                1.0,
                '/',
                '*',
                ('i2', None),
                ('i1', None),
                2.0
            ],
            'o3': [
                'switch',
                ('i2', None),
                '@high',
                '+',
                '+',
                ('i1', None),
                3.0,
                'Normal',
                0.0,
                '/',
                '*',
                ('i1', None),
                ('i1', None),
                4.0,
                '@medium',
                '+',
                '+',
                '-',
                ('i2', None),
                2,
                'Normal',
                1.0,
                '/',
                '*',
                ('i2', None),
                ('i1', None),
                2.0,
                'default',
                '*',
                '-',
                'Normal',
                1.0,
                0.0,
                '-',
                16.0
            ],
            "rlevel2'": [
                'sum',
                ('?up', 'res'),
                '*',
                ('DOWNSTREAM', ['?up', '?r']),
                '+',
                ('outflow', ['?up']),
                ('overflow', ['?up'])
            ],
            "rlevel3'": [
                '+',
                '-',
                '-',
                '-',
                '+',
                ('rlevel', ['?r']),
                ('rainfall', ['?r']),
                ('evaporated', ['?r']),
                ('outflow', ['?r']),
                ('overflow', ['?r']),
                'sum',
                ('?up', 'res'),
                '*',
                ('DOWNSTREAM', ['?up', '?r']),
                '+',
                ('outflow', ['?up']),
                ('overflow', ['?up'])
            ],
            "rlevel4'": [
                '+',
                '-',
                '-',
                '+',
                ('rlevel', ['?r']),
                ('rainfall', ['?r']),
                ('evaporated', ['?r']),
                ('outflow', ['?r']),
                'sum',
                ('?up', 'res'),
                '-',
                '*',
                ('DOWNSTREAM', ['?up', '?r']),
                '+',
                ('outflow', ['?up']),
                ('overflow', ['?up']),
                ('overflow', ['?r'])
            ],
            "rlevel5'": [
                '-',
                '+',
                '-',
                '-',
                '+',
                ('rlevel', ['?r']),
                ('rainfall', ['?r']),
                ('evaporated', ['?r']),
                ('outflow', ['?r']),
                'sum',
                ('?up', 'res'),
                '*',
                ('DOWNSTREAM', ['?up', '?r']),
                '+',
                ('outflow', ['?up']),
                ('overflow', ['?up']),
                ('overflow', ['?r'])
            ],
            "rlevel6'": [
                'max',
                ('?up', 'res'),
                ('?down', 'res2'),
                '+',
                '*',
                ('DOWNSTREAM', ['?up', '?down']),
                ('outflow', ['?up']),
                ('overflow', ['?up'])
            ],
            "rlevel7'": [
                '+',
                '-',
                '-',
                '+',
                ('rlevel', ['?r']),
                ('rainfall', ['?r']),
                ('evaporated', ['?r']),
                ('outflow', ['?r']),
                'sum',
                ('?up', 'res1'),
                ('?down', 'res'),
                '-',
                '*',
                ('DOWNSTREAM', ['?up', '?down']),
                '+',
                ('outflow', ['?up']),
                ('overflow', ['?up']),
                ('overflow', ['?r'])
            ],
            "hintedRight'": [
                'KronDelta',
                '^',
                '^',
                'forall',
                ('?s3', 'skill'),
                '~',
                ('updateTurn', ['?s3']),
                ('giveHint', ['?s']),
                'forall',
                ('?s2', 'skill'),
                '=>',
                ('PRE_REQ', ['?s2', '?s']),
                ('proficiencyHigh', ['?s2'])
            ],
            "hintDelayVar'": [
                'KronDelta',
                '^',
                'forall',
                ('?s2', 'skill'),
                '~',
                ('updateTurn', ['?s2']),
                ('giveHint', ['?s'])
            ],
            "robot-at'": [
                'if',
                'exists',
                ('?x2', 'xpos'),
                ('?y2', 'ypos'),
                '^',
                ('GOAL', ['?x2', '?y2']),
                ('robot-at', ['?x2', '?y2']),
                'KronDelta',
                False,
                'if',
                '^',
                ('move-north', None),
                'exists',
                ('?y2', 'ypos'),
                '^',
                '^',
                ('NORTH', ['?y2', '?y']),
                ('robot-at', ['?x', '?y2']),
                '~',
                ('obstacle-at', ['?x', '?y2']),
                'KronDelta',
                True,
                False
            ]
        }

        for cpf in cpfs:

            pvar = cpf.pvar[1][0]
            self.assertIn(pvar, ast)

            expected = ast[pvar]
            i = 0

            stack = [cpf.expr]
            while len(stack) > 0:
                expr = stack.pop()
                if expr[0] == 'pvar_expr':
                    self.assertEqual(expr[1], expected[i])
                elif expr[0] == 'number':
                    if isinstance(expr[1], int):
                        self.assertEqual(expr[1], expected[i])
                    else:
                        self.assertAlmostEqual(expr[1], expected[i])
                elif expr[0] == 'boolean':
                    self.assertEqual(expr[1], expected[i])
                elif expr[0] == 'func':
                    self.assertEqual(expr[1][0], expected[i])
                    for subexpr in expr[1][1][::-1]:
                        stack.append(subexpr)
                elif expr[0] == 'enum_type':
                    self.assertEqual(expr[1], expected[i])
                elif expr[0] == 'typed_var':
                    self.assertEqual(expr[1], expected[i])
                elif expr[0] == 'lconst':
                    self.assertEqual(expr[1][0], expected[i])
                    stack.append(expr[1][1])
                elif expr[0] == 'case':
                    self.assertEqual(expr[1][0], expected[i])
                    stack.append(expr[1][1])
                elif expr[0] == 'default':
                    self.assertEqual(expr[0], expected[i])
                    stack.append(expr[1])
                elif expr[0] == 'randomvar':
                    self.assertEqual(expr[1][0], expected[i])
                    for subexpr in expr[1][1][::-1]:
                        stack.append(subexpr)
                else:
                    self.assertEqual(expr[0], expected[i])
                    for subexpr in expr[1][::-1]:
                        stack.append(subexpr)
                i += 1

    def test_instance_block(self):
        instance = self.rddl.instance
        self.assertIsInstance(instance, Instance)
        self.assertEqual(instance.name, 'inst_reservoir_res8')

    def test_nonfluents_block(self):
        non_fluents = self.rddl.non_fluents
        self.assertIsInstance(non_fluents, NonFluents)
        self.assertEqual(non_fluents.name, 'res8')
