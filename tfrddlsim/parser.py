from ply import lex, yacc

from tfrddlsim.rddl import RDDL, Domain, Instance, NonFluents
from tfrddlsim.pvariable import NonFluent, StateFluent, ActionFluent, IntermediateFluent
from tfrddlsim.cpf import CPF


alpha = r'[A-Za-z]'
digit = r'[0-9]'
idenfifier = r'(' + alpha + r')((' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r'))?(\')?'
integer = digit + r'+'
double = digit + r'*\.' + digit + r'+'
variable = r'\?(' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r')'
enum_value = r'\@(' + alpha + r'|' + digit + r'|\-|\_)*(' + alpha + r'|' + digit + r')'


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
            'ENUM_VAL',
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

    @lex.TOKEN(enum_value)
    def t_ENUM_VAL(self, t):
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

        self.precedence = (
            ('left', 'IF'),
            ('left', 'ASSIGN_EQUAL'),
            ('left', 'EXISTS'),
            ('left', 'FORALL'),
            ('left', 'AGG_OPER'),
            ('left', 'EQUIV'),
            ('left', 'IMPLY'),
            ('left', 'OR'),
            ('left', 'AND', 'AMPERSAND'),
            ('left', 'NOT'),
            ('left', 'COMP_EQUAL', 'NEQ', 'LESS', 'LESSEQ', 'GREATER', 'GREATEREQ'),
            ('left', 'PLUS', 'MINUS'),
            ('left', 'TIMES', 'DIV'),
            ('right', 'UMINUS')
        )

    def p_rddl(self, p):
        '''rddl : rddl_block'''
        p[0] = RDDL(p[1])

    def p_rddl_block(self, p):
        '''rddl_block : rddl_block domain_block
                      | rddl_block instance_block
                      | rddl_block nonfluent_block
                      | empty'''
        if p[1] is None:
            p[0] = dict()
        else:
            name, block = p[2]
            p[1][name] = block
            p[0] = p[1]

    def p_domain_block(self, p):
        '''domain_block : DOMAIN IDENT LCURLY req_section domain_list RCURLY'''
        d = Domain(name=p[2], requirements=p[4], domain_list=p[5])
        p[0] = ('domain', d)

    def p_req_section(self, p):
        '''req_section : REQUIREMENTS ASSIGN_EQUAL LCURLY string_list RCURLY SEMI
                       | REQUIREMENTS LCURLY string_list RCURLY SEMI
                       | empty'''
        if len(p) == 7:
            p[0] = p[4]
        elif len(p) == 6:
            p[0] = p[3]

    def p_domain_list(self, p):
        '''domain_list : domain_list type_section
                       | domain_list pvar_section
                       | domain_list cpf_section
                       | empty'''
        if p[1] is None:
            p[0] = dict()
        else:
            name, section = p[2]
            p[1][name] = section
            p[0] = p[1]

    def p_type_section(self, p):
        '''type_section : TYPES LCURLY type_list RCURLY SEMI'''
        p[0] = ('types', p[3])

    def p_type_list(self, p):
        '''type_list : type_list type_def
                     | empty'''
        if p[1] is None:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_type_def(self, p):
        '''type_def : IDENT COLON OBJECT SEMI
                    | IDENT COLON LCURLY enum_list RCURLY SEMI'''
        if len(p) == 5:
            p[0] = (p[1], p[3])
        elif len(p) == 7:
            p[0] = (p[1], p[4])

    def p_enum_list(self, p):
        '''enum_list : enum_list COMMA ENUM_VAL
                     | ENUM_VAL
                     | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_pvar_section(self, p):
        '''pvar_section : PVARIABLES LCURLY pvar_list RCURLY SEMI'''
        p[0] = ('pvariables', p[3])

    def p_pvar_list(self, p):
        '''pvar_list : pvar_list pvar_def
                     | empty'''
        if p[1] is None:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_pvar_def(self, p):
        '''pvar_def : nonfluent_def
                    | statefluent_def
                    | actionfluent_def
                    | intermfluent_def'''
        p[0] = p[1]

    def p_nonfluent_def(self, p):
        '''nonfluent_def : IDENT LPAREN param_list RPAREN COLON LCURLY NON_FLUENT COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI
                         | IDENT COLON LCURLY NON_FLUENT COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI'''
        if len(p) == 16:
            p[0] = NonFluent(name=p[1], range_type=p[9], param_types=p[3], def_value=p[13])
        else:
            p[0] = NonFluent(name=p[1], range_type=p[6], def_value=p[10])

    def p_statefluent_def(self, p):
        '''statefluent_def : IDENT LPAREN param_list RPAREN COLON LCURLY STATE COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI
                           | IDENT COLON LCURLY STATE COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI'''
        if len(p) == 16:
            p[0] = StateFluent(name=p[1], range_type=p[9], param_types=p[3], def_value=p[13])
        else:
            p[0] = StateFluent(name=p[1], range_type=p[6], def_value=p[10])

    def p_actionfluent_def(self, p):
        '''actionfluent_def : IDENT LPAREN param_list RPAREN COLON LCURLY ACTION COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI
                            | IDENT COLON LCURLY ACTION COMMA type_spec COMMA DEFAULT ASSIGN_EQUAL range_const RCURLY SEMI'''
        if len(p) == 16:
            p[0] = ActionFluent(name=p[1], range_type=p[9], param_types=p[3], def_value=p[13])
        else:
            p[0] = ActionFluent(name=p[1], range_type=p[6], def_value=p[10])

    def p_intermfluent_def(self, p):
        '''intermfluent_def : IDENT LPAREN param_list RPAREN COLON LCURLY INTERMEDIATE COMMA type_spec COMMA LEVEL ASSIGN_EQUAL range_const RCURLY SEMI
                            | IDENT COLON LCURLY INTERMEDIATE COMMA type_spec COMMA LEVEL ASSIGN_EQUAL range_const RCURLY SEMI'''
        if len(p) == 16:
            p[0] = IntermediateFluent(name=p[1], range_type=p[9], level=p[13], param_types=p[3])
        else:
            p[0] = IntermediateFluent(name=p[1], range_type=p[6], level=p[10])

    def p_cpf_section(self, p):
        '''cpf_section : cpf_header LCURLY cpf_list RCURLY SEMI'''
        p[0] = ('cpfs', (p[1], p[3]))

    def p_cpf_header(self, p):
        '''cpf_header : CPFS
                      | CDFS'''
        p[0] = p[1]

    def p_cpf_list(self, p):
        '''cpf_list : cpf_list cpf_def
                    | empty'''
        if p[1] is None:
            p[0] = []
        else:
            p[1].append(p[2])
            p[0] = p[1]

    def p_cpf_def(self, p):
        '''cpf_def : pvar_expr ASSIGN_EQUAL expr SEMI'''
        p[0] = CPF(pvar=p[1], expr=p[3])

    def p_term_list(self, p):
        '''term_list : term_list COMMA term
                     | term
                     | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_term(self, p):
        '''term : VAR
                | ENUM_VAL
                | pvar_expr'''
        p[0] = p[1]

    def p_expr(self, p):
        '''expr : pvar_expr
                | group_expr
                | function_expr
                | relational_expr
                | boolean_expr
                | quantifier_expr
                | numerical_expr
                | aggregation_expr
                | control_expr
                | randomvar_expr'''
        p[0] = p[1]

    def p_pvar_expr(self, p):
        '''pvar_expr : IDENT LPAREN term_list RPAREN
                     | IDENT'''
        if len(p) == 2:
            p[0] = ('pvar_expr', (p[1], None))
        elif len(p) == 5:
            p[0] = ('pvar_expr', (p[1], p[3]))

    def p_group_expr(self, p):
        '''group_expr : LBRACK expr RBRACK
                      | LPAREN expr RPAREN'''
        p[0] = p[2]

    def p_function_expr(self, p):
        '''function_expr : IDENT LBRACK expr_list RBRACK'''
        p[0] = ('func', (p[1], p[3]))

    def p_relational_expr(self, p):
        '''relational_expr : expr COMP_EQUAL expr
                           | expr NEQ expr
                           | expr GREATER expr
                           | expr GREATEREQ expr
                           | expr LESS expr
                           | expr LESSEQ expr'''
        p[0] = (p[2], (p[1], p[3]))

    def p_boolean_expr(self, p):
        '''boolean_expr : expr AND expr
                        | expr AMPERSAND expr
                        | expr OR expr
                        | expr IMPLY expr
                        | expr EQUIV expr
                        | NOT expr %prec UMINUS
                        | bool_type'''
        if len(p) == 4:
            p[0] = (p[2], (p[1], p[3]))
        elif len(p) == 3:
            p[0] = (p[1], (p[2],))
        elif len(p) == 2:
            p[0] = ('boolean', p[1])

    def p_quantifier_expr(self, p):
        '''quantifier_expr : FORALL UNDERSCORE LCURLY typed_var_list RCURLY expr %prec FORALL
                           | EXISTS UNDERSCORE LCURLY typed_var_list RCURLY expr %prec EXISTS'''
        p[0] = (p[1], (*p[4], p[6]))

    def p_numerical_expr(self, p):
        '''numerical_expr : expr PLUS expr
                          | expr MINUS expr
                          | expr TIMES expr
                          | expr DIV expr
                          | MINUS expr %prec UMINUS
                          | PLUS expr %prec UMINUS
                          | INTEGER
                          | DOUBLE'''
        if len(p) == 4:
            p[0] = (p[2], (p[1], p[3]))
        elif len(p) == 3:
            p[0] = (p[1], (p[2],))
        elif len(p) == 2:
            p[0] = ('number', p[1])

    def p_aggregation_expr(self, p):
        '''aggregation_expr : IDENT UNDERSCORE LCURLY typed_var_list RCURLY expr %prec AGG_OPER'''
        p[0] = (p[1], (*p[4], p[6]))

    def p_control_expr(self, p):
        '''control_expr : IF LPAREN expr RPAREN THEN expr ELSE expr %prec IF
                        | SWITCH LPAREN term RPAREN LCURLY case_list RCURLY'''
        if len(p) == 9:
            p[0] = (p[1], (p[3], p[6], p[8]))
        elif len(p) == 8:
            p[0] = (p[1], (p[3], *p[6]))

    def p_randomvar_expr(self, p):
        '''randomvar_expr : BERNOULLI LPAREN expr RPAREN
                          | DIRAC_DELTA LPAREN expr RPAREN
                          | KRON_DELTA LPAREN expr RPAREN
                          | UNIFORM LPAREN expr COMMA expr RPAREN
                          | NORMAL LPAREN expr COMMA expr RPAREN
                          | EXPONENTIAL LPAREN expr RPAREN
                          | DISCRETE LPAREN IDENT COMMA lconst_case_list RPAREN
                          | DIRICHLET LPAREN IDENT COMMA expr RPAREN
                          | POISSON LPAREN expr RPAREN
                          | WEIBULL LPAREN expr COMMA expr RPAREN
                          | GAMMA   LPAREN expr COMMA expr RPAREN'''
        if len(p) == 7:
            if isinstance(p[5], list):
                p[0] = ('randomvar', (p[1], (('enum_type', p[3]), *p[5])))
            else:
                p[0] = ('randomvar', (p[1], (p[3], p[5])))
        elif len(p) == 5:
            p[0] = ('randomvar', (p[1], (p[3],)))

    def p_typed_var_list(self, p):
        '''typed_var_list : typed_var_list COMMA typed_var
                          | typed_var'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_typed_var(self, p):
        '''typed_var : VAR COLON IDENT'''
        p[0] = ('typed_var', (p[1], p[3]))

    def p_expr_list(self, p):
        '''expr_list : expr_list COMMA expr
                     | expr'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_case_list(self, p):
        '''case_list : case_list COMMA case_def
                     | case_def'''
        if len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_case_def(self, p):
        '''case_def : CASE term COLON expr
                    | DEFAULT COLON expr'''
        if len(p) == 5:
            p[0] = ('case', (p[2], p[4]))
        elif len(p) == 4:
            p[0] = ('default', p[3])

    def p_lconst_case_list(self, p):
        '''lconst_case_list : lconst COLON expr
                            | lconst COLON OTHERWISE
                            | lconst_case_list COMMA lconst COLON expr'''
        if len(p) == 4:
            p[0] = [('lconst', (p[1], p[3]))]
        elif len(p) == 6:
            p[1].append(('lconst', (p[3], p[5])))
            p[0] = p[1]

    def p_lconst(self, p):
        '''lconst : IDENT
                  | ENUM_VAL'''
        p[0] = p[1]

    def p_param_list(self, p):
        '''param_list : string_list'''
        p[0] = p[1]

    def p_type_spec(self, p):
        '''type_spec : IDENT
                     | INT
                     | REAL
                     | BOOL'''
        p[0] = p[1]

    def p_range_const(self, p):
        '''range_const : bool_type
                       | double_type
                       | int_type
                       | IDENT'''
        p[0] = p[1]

    def p_bool_type(self, p):
        '''bool_type : TRUE
                     | FALSE'''
        p[0] = True if p[1] == 'true' else False

    def p_double_type(self, p):
        '''double_type : DOUBLE
                       | MINUS DOUBLE
                       | POS_INF
                       | NEG_INF'''
        p[0] = p[1] if len(p) == 2 else -p[2]

    def p_int_type(self, p):
        '''int_type : INTEGER
                    | MINUS INTEGER'''
        p[0] = p[1] if len(p) == 2 else -p[2]

    def p_instance_block(self, p):
        '''instance_block : INSTANCE IDENT LCURLY RCURLY'''
        i = Instance(p[2])
        p[0] = ('instance', i)

    def p_nonfluent_block(self, p):
        '''nonfluent_block : NON_FLUENTS IDENT LCURLY RCURLY'''
        nf = NonFluents(p[2])
        p[0] = ('non_fluents', nf)

    def p_string_list(self, p):
        '''string_list : string_list COMMA IDENT
                       | IDENT
                       | empty'''
        if p[1] is None:
            p[0] = []
        elif len(p) == 4:
            p[1].append(p[3])
            p[0] = p[1]
        elif len(p) == 2:
            p[0] = [p[1]]

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        print('Syntax error in input!')

    def build(self, **kwargs):
        self._parser = yacc.yacc(module=self, **kwargs)

    def parse(self, input):
        return self._parser.parse(input=input, lexer=self.lexer)
