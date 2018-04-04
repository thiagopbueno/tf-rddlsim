from ply import lex, yacc

from tfrddlsim.rddl import RDDL, Domain, Instance, NonFluents


alpha = r'[A-Za-z]'
digit = r'[0-9]'
idenfifier = r'(' + alpha + r')((' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r'))?(\')?'
integer = digit + r'+'
double = digit + r'*\.' + digit + r'+'
variable = r'\?(' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r')'


class RDDLlex(object):

    def __init__(self):
        self.reserved = {
            'domain': 'DOMAIN',
            'instance': 'INSTANCE',
            'horizon': 'HORIZON',
            'discount': 'DISCOUNT',
            'objects': 'OBJECTS',
            'init-state': 'INIT_STATE',
            'requirements': 'REQUIREMENTS',
            'state-action-constraints': 'STATE_ACTION_CONSTRAINTS',
            'action-preconditions': 'ACTION_PRECONDITIONS',
            'state-invariants': 'STATE_INVARIANTS',
            'types': 'TYPES',
            'object': 'OBJECT',
            'bool': 'BOOL',
            'int': 'INT',
            'real': 'REAL',
            'neg-inf': 'NEG_INF',
            'pos-inf': 'POS_INF',
            'pvariables': 'PVARIABLES',
            'non-fluent': 'NON_FLUENT',
            'non-fluents': 'NON_FLUENTS',
            'state-fluent': 'STATE',
            'interm-fluent': 'INTERMEDIATE',
            'derived-fluent': 'DERIVED_FLUENT',
            'observ-fluent': 'OBSERVATION',
            'action-fluent': 'ACTION',
            'level': 'LEVEL',
            'default': 'DEFAULT',
            'max-nondef-actions': 'MAX_NONDEF_ACTIONS',
            'terminate-when': 'TERMINATE_WHEN',
            'terminal': 'TERMINAL',
            'cpfs': 'CPFS',
            'cdfs': 'CDFS',
            'reward': 'REWARD',
            'forall': 'FORALL',
            'exists': 'EXISTS',
            'true': 'TRUE',
            'false': 'FALSE',
            'if': 'IF',
            'then': 'THEN',
            'else': 'ELSE',
            'switch': 'SWITCH',
            'case': 'CASE',
            'otherwise': 'OTHERWISE',
            'KronDelta': 'KRON_DELTA',
            'DiracDelta': 'DIRAC_DELTA',
            'Uniform': 'UNIFORM',
            'Bernoulli': 'BERNOULLI',
            'Discrete': 'DISCRETE',
            'Normal': 'NORMAL',
            'Poisson': 'POISSON',
            'Exponential': 'EXPONENTIAL',
            'Weibull': 'WEIBULL',
            'Gamma': 'GAMMA',
            'Multinomial': 'MULTINOMIAL',
            'Dirichlet': 'DIRICHLET'
        }

        self.tokens = [
            'IDENT',
            'VAR',
            'INTEGER',
            'DOUBLE',
            'AND',
            'OR',
            'NOT',
            'PLUS',
            'TIMES',
            'LPAREN',
            'RPAREN',
            'LCURLY',
            'RCURLY',
            'DOT',
            'COMMA',
            'UNDERSCORE',
            'LBRACK',
            'RBRACK',
            'IMPLY',
            'EQUIV',
            'NEQ',
            'LESSEQ',
            'LESS',
            'GREATEREQ',
            'GREATER',
            'ASSIGN_EQUAL',
            'COMP_EQUAL',
            'DIV',
            'MINUS',
            'COLON',
            'SEMI',
            'DOLLAR_SIGN',
            'QUESTION',
            'AMPERSAND'
        ]
        self.tokens += list(self.reserved.values())

    t_ignore = ' \t'

    t_AND = r'\^'
    t_OR = r'\|'
    t_NOT = r'~'
    t_PLUS = r'\+'
    t_TIMES = r'\*'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LCURLY = r'\{'
    t_RCURLY = r'\}'
    t_DOT = r'\.'
    t_COMMA = r'\,'
    t_UNDERSCORE = r'\_'
    t_LBRACK = r'\['
    t_RBRACK = r'\]'
    t_IMPLY = r'=>'
    t_EQUIV = r'<=>'
    t_NEQ = r'~='
    t_LESSEQ = r'<='
    t_LESS = r'<'
    t_GREATEREQ = r'>='
    t_GREATER = r'>'
    t_ASSIGN_EQUAL = r'='
    t_COMP_EQUAL = r'=='
    t_DIV = r'/'
    t_MINUS = r'-'
    t_COLON = r':'
    t_SEMI = r';'
    t_DOLLAR_SIGN = r'\$'
    t_QUESTION = r'\?'
    t_AMPERSAND = r'\&'

    def t_newline(self, t):
        r'\n+'
        self._lexer.lineno += len(t.value)

    def t_COMMENT(self, t):
        r'//[^\r\n]*'
        pass

    @lex.TOKEN(idenfifier)
    def t_IDENT(self, t):
        t.type = self.reserved.get(t.value, 'IDENT')
        return t

    @lex.TOKEN(variable)
    def t_VAR(self, t):
        return t

    @lex.TOKEN(double)
    def t_DOUBLE(self, t):
        t.value = float(t.value)
        return t

    @lex.TOKEN(integer)
    def t_INTEGER(self, t):
        t.value = int(t.value)
        return t

    def t_error(self, t):
        print("Illegal character: {} at line {}".format(t.value[0], self._lexer.lineno))
        t.lexer.skip(1)

    def build(self, **kwargs):
        self._lexer = lex.lex(object=self, **kwargs)

    def input(self, data):
        if self._lexer is None:
            self.build()
        self._lexer.input(data)

    def token(self):
        return self._lexer.token()

    def __call__(self):
        while True:
            tok = self.token()
            if not tok:
                break
            yield tok


class RDDLParser(object):

    def __init__(self, lexer=None):
        if lexer is None:
            self.lexer = RDDLlex()
            self.lexer.build()

        self.tokens = self.lexer.tokens

    def p_rddl_block(self, p):
        '''rddl_block : domain_block rddl_block
                      | instance_block rddl_block
                      | nonfluent_block rddl_block
                      | empty'''
        if p[1] is None:
            p[0] = RDDL()
        else:
            p[2].add_block(p[1])
            p[0] = p[2]

    def p_domain_block(self, p):
        '''domain_block : DOMAIN IDENT LCURLY RCURLY'''
        p[0] = Domain(p[2])

    def p_instance_block(self, p):
        '''instance_block : INSTANCE IDENT LCURLY RCURLY'''
        p[0] = Instance(p[2])

    def p_nonfluent_block(self, p):
        '''nonfluent_block : NON_FLUENTS IDENT LCURLY RCURLY'''
        p[0] = NonFluents(p[2])

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        print('Syntax error in input!')

    def build(self, **kwargs):
        self._parser = yacc.yacc(module=self, **kwargs)

    def parse(self, input):
        return self._parser.parse(input=input, lexer=self.lexer)
