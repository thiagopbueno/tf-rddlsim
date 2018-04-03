from ply import lex


alpha = r'[A-Za-z]'
digit = r'[0-9]'
idenfifier = r'(' + alpha + r')((' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r'))?(\')?'


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

        self.tokens = ['ID'] + list(self.reserved.values())

    t_ignore = ' \t'

    def t_newline(self, t):
        r'\n+'
        self._lexer.lineno += len(t.value)

    def t_COMMENT(self, t):
        r'//[^\r\n]*'
        pass

    @lex.TOKEN(idenfifier)
    def t_ID(self, t):
        t.type = self.reserved.get(t.value, 'ID')
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

    def __call__(self):
        while True:
            tok = self._lexer.token()
            if not tok:
                break
            yield tok
