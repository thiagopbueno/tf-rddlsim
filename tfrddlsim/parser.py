from ply import lex


alpha = r'[A-Za-z]'
digit = r'[0-9]'
idenfifier = r'(' + alpha + r')((' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r'))?(\')?'


class RDDLlex(object):

    def __init__(self):
        self.tokens = ['ID']

    t_ignore = ' \t'

    def t_newline(self, t):
        r'\n+'
        self._lexer.lineno += len(t.value)

    def t_COMMENT(self, t):
        r'//[^\r\n]*'
        pass

    @lex.TOKEN(idenfifier)
    def t_ID(self, t):
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
