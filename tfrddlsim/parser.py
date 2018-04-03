from ply import lex

class RDDLlex(object):

    def __init__(self):
        self.tokens = ['None']

    t_ignore = ' \t'

    def t_newline(self, t):
        r'\n+'
        self._lexer.lineno += len(t.value)

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
