import ply.lex as lex
import ply.yacc as yacc
from dataclasses import dataclass, is_dataclass
from typing import Optional, Any, List, Tuple, Dict
import os
import matplotlib.pyplot as plt

#parte dos nós dataclass da ast
@dataclass
class Program:
    body: List[Any]

@dataclass
class FunctionDef:
    rettype: Any
    name: str
    params: List[Any]
    body: Any
    line: int
    col: int = 0

@dataclass
class Declaration:
    type: Any
    decls: List[Any]
    line: int
    col: int = 0

@dataclass
class VarDecl:
    name: str
    init: Optional[Any]
    line: int
    col: int = 0

@dataclass
class Assign:
    target: Any
    value: Any
    line: int
    col: int = 0

@dataclass
class If:
    test: Any
    then: Any
    otherwise: Optional[Any]
    line: int
    col: int = 0

@dataclass
class While:
    test: Any
    body: Any
    line: int
    col: int = 0

@dataclass
class For:
    init: Optional[Any]
    cond: Optional[Any]
    post: Optional[Any]
    body: Any
    line: int
    col: int = 0

@dataclass
class Block:
    body: List[Any]
    line: int
    col: int = 0

@dataclass
class Return:
    value: Optional[Any]
    line: int
    col: int = 0

@dataclass
class Call:
    callee: Any
    args: List[Any]
    line: int
    col: int = 0

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    line: int
    col: int = 0

@dataclass
class UnaryOp:
    op: str
    operand: Any
    line: int
    col: int = 0

@dataclass
class Var:
    name: str
    line: int
    col: int = 0

@dataclass
class Num:
    value: str
    line: int
    col: int = 0

@dataclass
class Str:
    value: str
    line: int
    col: int = 0

@dataclass
class Bool:
    value: bool
    line: int
    col: int = 0

#parte do lexer
keywords = {
    'int': 'INT',
    'float': 'FLOAT',
    'if': 'IF',
    'else': 'ELSE',
    'for': 'FOR',
    'while': 'WHILE',
    'return': 'RETURN',
    'void': 'VOID',
    'char': 'CHAR',
    'double': 'DOUBLE',
    'true': 'TRUE',
    'false': 'FALSE',
}

tokens = [
    'ID', 'INT_CONST', 'FLOAT_CONST', 'STRING_LITERAL', 'CHAR_CONST',
    'PLUS', 'MINUS', 'STAR', 'SLASH', 'PERCENT',
    'EQEQ', 'NE', 'LT', 'LE', 'GT', 'GE',
    'ASSIGN', 'PLUSEQ', 'MINUSEQ',
    'AND', 'OR', 'NOT',
    'PLUSPLUS', 'MINUSMINUS',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'SEMI', 'COMMA', 'LBRACK', 'RBRACK',
    'PP_DIRECTIVE',
] + list(keywords.values())

t_PLUS          = r'\+'
t_MINUS         = r'-'
t_STAR          = r'\*'
t_SLASH         = r'/'
t_PERCENT       = r'%'
t_ASSIGN        = r'='
t_PLUSEQ        = r'\+='
t_MINUSEQ       = r'-='
t_EQEQ          = r'=='
t_NE            = r'!='
t_LT            = r'<'
t_LE            = r'<='
t_GT            = r'>'
t_GE            = r'>='
t_AND           = r'&&'
t_OR            = r'\|\|'
t_NOT           = r'!'
t_PLUSPLUS      = r'\+\+'
t_MINUSMINUS    = r'--'
t_LPAREN        = r'\('
t_RPAREN        = r'\)'
t_LBRACE        = r'{'
t_RBRACE        = r'}'
t_LBRACK        = r'\['
t_RBRACK        = r'\]'
t_SEMI          = r';'
t_COMMA         = r','

def t_STRING_LITERAL(t):
    r'\"([^\\\n]|\\.)*?\"'
    return t

def t_CHAR_CONST(t):
    r'\'([^\\\n]|\\.)\''
    return t

def t_FLOAT_CONST(t):
    r'(\d+\.\d*([Ee][+-]?\d+)?)|(\.\d+([Ee][+-]?\d+)?)|(\d+([Ee][+-]?\d+))'
    return t

def t_INT_CONST(t):
    r'\d+'
    return t

def t_PP_DIRECTIVE(t):
    r'\#(.*)'
    pass

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    t.type = keywords.get(t.value, 'ID')
    return t

def t_COMMENT(t):
    r'/\*(.|\n)*?\*/ | //.*'
    pass

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

t_ignore  = ' \t'

def t_error(t):
    print(f"Erro Léxico: Caractere ilegal '{t.value[0]}' na linha {t.lineno}")
    t.lexer.skip(1)

lexer = lex.lex()


#parte da visualização da ast
def node_label(n: Any) -> str:
    if n is None: return "None"
    tname = type(n).__name__
    
    if tname == "Program": return "Program"
    if tname == "FunctionDef": return f"Fnc({getattr(n,'name','?')})"
    if tname == "Declaration": return f"Typ({getattr(n,'type','?')})"
    if tname == "VarDecl": return f"Var({getattr(n,'name','?')})"
    if tname == "Assign": return "Assign"
    if tname == "If": return "If"
    if tname == "For": return "For"
    if tname == "Return": return "Return"
    if tname == "Block": return "Block"
    if tname == "Call": return "Call"
    if tname == "BinOp": return f"BinOp('{getattr(n, 'op', '?')}')"
    if tname == "UnaryOp": return f"Unary('{getattr(n,'op','?')}')"
    if tname == "Var": return f"Id({getattr(n, 'name', '?')})"
    if tname == "Num": return f"Num({getattr(n, 'value', '?')})"
    if tname == "Str": return f"Str({getattr(n, 'value', '?')})"
    if tname == "Bool":
        v = getattr(n, "value", None)
        return f"Bool({str(v).lower()})" if isinstance(v, bool) else "Bool(?)"
    
    return tname

def get_children(n: Any) -> List[Any]:
    if n is None: return []
    tname = type(n).__name__
    
    if tname == "Program": return list(getattr(n, "body", []))
    if tname == "FunctionDef":
        params = getattr(n, "params", []) or []
        return [getattr(n, "rettype", None), *params, getattr(n, "body", None)]
    if tname == "Declaration": return list(getattr(n, "decls", []))
    if tname == "VarDecl": return [getattr(n, "init", None)]
    if tname == "Assign": return [getattr(n, "target", None), getattr(n, "value", None)]
    if tname == "If":
        lst = [getattr(n, "test", None), getattr(n, "then", None)]
        other = getattr(n, "otherwise", None)
        if other is not None: lst.append(other)
        return lst
    
    if tname == "While":
        return [getattr(n, "test", None), getattr(n, "body", None)] 

    if tname == "For":
        return [getattr(n, "init", None), getattr(n, "cond", None),
                getattr(n, "post", None), getattr(n, "body", None)]
    if tname == "Return": return [getattr(n, "value", None)]
    if tname == "Block": return list(getattr(n, "body", []))
    if tname == "Call": return [getattr(n, "callee", None)] + list(getattr(n, "args", []))
    if tname == "BinOp": return [getattr(n, "left", None), getattr(n, "right", None)]
    if tname == "UnaryOp": return [getattr(n, "operand", None)]

    if is_dataclass(n):
        out: List[Any] = []
        for k, v in n.__dict__.items():
            if k in ("line", "col", "op", "name", "value", "rettype", "type", "decls", "target", "test", "then", "otherwise", "init", "cond", "post", "body", "left", "right", "operand", "callee", "args", "params"):
                continue
            if isinstance(v, list): out.extend(v)
            elif v is not None: out.append(v)
        return out
    
    if isinstance(n, list): return [x for x in n if x is not None]
    return []

def _compute_layout(n: Any, x0=0.0, y0=0.0, y_spacing=1.6) -> Tuple[Dict[int,Tuple[float,float]], float]:
    ch = [c for c in get_children(n) if c is not None]
    if not ch: return ({id(n): (x0, y0)}, 1.0)
    pos: Dict[int,Tuple[float,float]] = {}
    widths: List[float] = []
    subs: List[Any] = []
    for c in ch:
        subpos, w = _compute_layout(c, 0, 0, y_spacing)
        pos.update(subpos)
        widths.append(w)
        subs.append(c)
    total_w = sum(widths) + (len(widths)-1)*0.8
    cur_x = x0 - total_w/2.0
    def shift(node: Any, dx: float, dy: float):
        x, y = pos[id(node)]
        pos[id(node)] = (x + dx, y + dy)
        for cc in get_children(node):
            if cc is not None: shift(cc, dx, dy)
    for c, w in zip(subs, widths):
        cx = cur_x + w/2.0
        shift(c, cx, y0 + y_spacing)
        cur_x += w + 0.8
    pos[id(n)] = (x0, y0)
    return pos, total_w

def draw_tree(root: Any, filename: str, figsize=(10, 7), dpi: int = 160):
    pos, _ = _compute_layout(root, 0.0, 0.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    
    def draw_edges(node: Any):
        x, y = pos[id(node)]
        for c in get_children(node):
            if c is None: continue
            xc, yc = pos[id(c)]
            ax.plot([x, xc], [y-0.05, yc+0.05], lw=1, color='gray') 
            draw_edges(c)
    
    def draw_nodes(node: Any):
        x, y = pos[id(node)]
        bbox = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=1)
        ax.text(x, y, node_label(node), ha="center", va="center", bbox=bbox, fontsize=9)
        for c in get_children(node):
            if c is not None: draw_nodes(c)
    
    draw_edges(root)
    draw_nodes(root)
    
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    pad = 1.2
    ax.set_xlim(min(xs)-pad, max(xs)+pad)
    ax.set_ylim(min(ys)-pad, max(ys)+pad)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f" AST salva em: {filename}")

#parte do parser
precedence = (
    ('left', 'OR'),
    ('left', 'AND'),
    ('left', 'EQEQ', 'NE'),
    ('left', 'LT', 'LE', 'GT', 'GE'),
    ('left', 'PLUS', 'MINUS'),
    ('left', 'STAR', 'SLASH', 'PERCENT'),
    ('right', 'NOT', 'PLUSPLUS', 'MINUSMINUS'),
)

def p_program(p):
    """program : external_decls"""
    items = [x for x in (p[1] or []) if x is not None]
    p[0] = Program(body=items)

def p_external_decls_multi(p):
    """external_decls : external_decls external_decl
                      | external_decl"""
    if len(p) == 3:
        left = p[1] or []
        p[0] = left + ([p[2]] if p[2] is not None else [])
    else:
        p[0] = [p[1]] if p[1] is not None else []

def p_external_decl_ignore(p):
    """external_decl : PP_DIRECTIVE"""
    p[0] = None

def p_external_decl_func(p):
    "external_decl : TYPE ID LPAREN param_list_opt RPAREN compound_stmt"
    lineno = p.slice[2].lineno
    p[0] = FunctionDef(rettype=p[1], name=p[2], params=p[4], body=p[6], line=lineno)

def p_external_decl_decl(p):
    "external_decl : declaration"
    p[0] = p[1]

def p_type_spec(p):
    """TYPE : INT
            | FLOAT
            | VOID
            | CHAR
            | DOUBLE"""
    p[0] = p[1]

def p_param_list_opt(p):
    """param_list_opt : param_list
                      | VOID
                      | empty"""
    if len(p) == 2 and p[1] == 'void':
        p[0] = []
    else:
        p[0] = p[1] if p[1] is not None else []

def p_param_list_multi(p):
    """param_list : param_list COMMA param
                  | param"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_param(p):
    "param : TYPE ID"
    p[0] = VarDecl(name=p[2], init=None, line=p.slice[2].lineno)

def p_declaration(p):
    "declaration : TYPE init_decl_list SEMI"
    token_type = p.slice[1]
    decl_lineno = getattr(token_type, 'lineno', 0)
    
    t = p[1]
    inits = p[2] if p[2] is not None else []
    decls = [VarDecl(name=n, 
                     init=init, 
                     line=decl_lineno)
             for (n, init) in inits]
             
    p[0] = Declaration(type=t, decls=decls, line=decl_lineno)

def p_init_decl_list_multi(p):
    """init_decl_list : init_decl_list COMMA init_decl
                      | init_decl"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_init_decl_assign(p):
    "init_decl : ID ASSIGN expression"
    p[0] = (p[1], p[3])

def p_init_decl_id(p):
    "init_decl : ID"
    p[0] = (p[1], None)

def p_compound_stmt(p):
    "compound_stmt : LBRACE stmt_seq_opt RBRACE"
    seq = p[2] if p[2] is not None else []
    p[0] = Block(body=seq, line=p.slice[1].lineno)

def p_stmt_seq_opt(p):
    """stmt_seq_opt : stmt_seq
                    | empty"""
    p[0] = p[1] if p[1] is not None else []

def p_stmt_seq_multi(p):
    """stmt_seq : stmt_seq stmt
                | stmt"""
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    else:
        p[0] = [p[1]]

def p_stmt(p):
    """stmt : declaration
            | expression_stmt
            | if_stmt
            | while_stmt
            | for_stmt
            | return_stmt
            | compound_stmt"""
    p[0] = p[1]

def p_expression_stmt(p):
    """expression_stmt : expression SEMI
                       | SEMI"""
    if len(p) == 3:
        p[0] = p[1]
    else:
        p[0] = None 

def p_return_stmt(p):
    "return_stmt : RETURN expression SEMI"
    p[0] = Return(value=p[2], line=p.slice[1].lineno)

def p_if_stmt(p):
    """if_stmt : IF LPAREN expression RPAREN stmt
               | IF LPAREN expression RPAREN stmt ELSE stmt"""
    lineno = p.slice[1].lineno
    if len(p) == 6:
        p[0] = If(test=p[3], then=p[5], otherwise=None, line=lineno)
    else:
        p[0] = If(test=p[3], then=p[5], otherwise=p[7], line=lineno)

def p_while_stmt(p):
    "while_stmt : WHILE LPAREN expression RPAREN stmt"
    p[0] = While(test=p[3], body=p[5], line=p.slice[1].lineno)

def p_for_stmt(p):
    "for_stmt : FOR LPAREN for_init SEMI for_cond SEMI for_post RPAREN stmt"
    p[0] = For(init=p[3], cond=p[5], post=p[7], body=p[9], line=p.slice[1].lineno)

def p_for_init(p):
    """for_init : declaration"""
    p[0] = p[1]

def p_for_init_decl(p):
    "for_init : TYPE init_decl_list"
    t = p[1]
    inits = p[2] if p[2] is not None else []
    decls = [VarDecl(name=n, init=init, line=p.slice[1].lineno) for (n, init) in inits]
    p[0] = Declaration(type=t, decls=decls, line=p.slice[1].lineno)

def p_for_init_expr(p):
    "for_init : expression"
    p[0] = p[1]

def p_for_init_empty(p):
    "for_init : empty"
    p[0] = None

def p_for_cond(p):
    """for_cond : expression
                | empty"""
    p[0] = p[1]

def p_for_post(p):
    """for_post : expression
                | empty"""
    p[0] = p[1]

def p_expression_cast(p):
    """expression : LPAREN TYPE RPAREN expression %prec NOT"""
    lineno = p.slice[1].lineno
    p[0] = UnaryOp(op=f'(CAST: {p[2]})', operand=p[4], line=lineno)

def p_expression_assign(p):
    "expression : ID ASSIGN expression"
    lineno = p.slice[1].lineno
    p[0] = Assign(target=Var(name=p[1], line=lineno), value=p[3], line=lineno)

def p_expression_binop(p):
    """expression : expression PLUS expression
                  | expression MINUS expression
                  | expression STAR expression
                  | expression SLASH expression
                  | expression PERCENT expression
                  | expression EQEQ expression
                  | expression NE expression
                  | expression LT expression
                  | expression LE expression
                  | expression GT expression
                  | expression GE expression
                  | expression AND expression
                  | expression OR expression"""
    lineno = p.slice[2].lineno
    p[0] = BinOp(op=p[2], left=p[1], right=p[3], line=lineno)

def p_expression_unary(p):
    """expression : PLUS expression %prec NOT
                  | MINUS expression %prec NOT
                  | NOT expression"""
    lineno = p.slice[1].lineno
    p[0] = UnaryOp(op=p[1], operand=p[2], line=lineno)

def p_expression_postfix_incdec(p):
    """expression : expression PLUSPLUS
                  | expression MINUSMINUS"""
    lineno = p.slice[2].lineno
    p[0] = UnaryOp(op=p[2], operand=p[1], line=lineno)

def p_expression_call(p):
    "expression : ID LPAREN arg_list_opt RPAREN"
    lineno = p.slice[1].lineno
    p[0] = Call(callee=Var(name=p[1], line=lineno), args=p[3], line=lineno)

def p_arg_list_opt(p):
    """arg_list_opt : arg_list
                    | empty"""
    p[0] = p[1] if p[1] is not None else []

def p_arg_list_multi(p):
    """arg_list : arg_list COMMA expression
                | expression"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]

def p_expression_group(p):
    "expression : LPAREN expression RPAREN"
    p[0] = p[2]

def p_expression_term(p):
    """expression : INT_CONST
                  | FLOAT_CONST
                  | CHAR_CONST
                  | STRING_LITERAL
                  | TRUE
                  | FALSE
                  | ID"""
    
    token = p.slice[1]
    lineno = token.lineno
    
    if token.type in ['INT_CONST', 'FLOAT_CONST']:
        p[0] = Num(value=p[1], line=lineno)
    elif token.type in ['CHAR_CONST', 'STRING_LITERAL']:
        p[0] = Str(value=p[1], line=lineno)
    elif token.type in ['TRUE', 'FALSE']:
        p[0] = Bool(value=(token.type == 'TRUE'), line=lineno)
    elif token.type == 'ID':
        p[0] = Var(name=p[1], line=lineno)

def p_empty(p):
    "empty :"
    p[0] = None

sync_tokens = ('SEMI', 'RBRACE', 'IF', 'FOR', 'WHILE', 'RETURN', 'INT', 'FLOAT', 'VOID')

houveErro = False
def p_error(p):
    global houveErro 
    houveErro = True
    if not p:
        ultima_linha_processada = lexer.lineno - 1 if lexer.lineno > 1 else 1

        print("Erro sintático: Fim de arquivo inesperado (EOF).")
        print(f"    Diagnóstico: Estrutura não fechada (após a Linha {ultima_linha_processada}).")
        return
    
    if p.type == 'RBRACE' or p.type in sync_tokens:
        linha_erro_real = p.lineno - 1
        print(f"Erro sintático: Comando não finalizado na Linha {linha_erro_real}.")
        print(f"    Esperado: ';'. Encontrado: '{p.value}' na Linha {p.lineno}.")
    else:
        print(f"Erro sintático: Token '{p.value}' inesperado na linha {p.lineno}.")
        
    parser.errok() 
    
    while True:
        tok = parser.token()
        if not tok or tok.type in sync_tokens:
            break
    if tok and tok.type in ('SEMI', 'RBRACE'):
        parser.token()

parser = yacc.yacc()

#parte do teste
testes = [
    {
        "name": "caso_1_atribuicao_soma",
        "desc": "Atribuição simples com soma (y = 1 + 3;)",
        "code": """
        int main() {
            int y;
            y = 1 + 3;
        }
        """
    },
    {
        "name": "caso_2_expr_parenteses",
        "desc": "Expressão aritmética com parênteses (2 * (3 + 4))",
        "code": """
        int main() {
            int result;
            result = 2 * (3 + 4);
        }
        """
    },
    {
        "name": "caso_3_dois_stmts",
        "desc": "Duas instruções em sequência (x = 5; x / 2;)",
        "code": """
        int main() {
            int x;
            x = 5;
            x = x / 2;
        }
        """
    },
    {
        "name": "caso_4_if_else_logica",
        "desc": "if/else com && e != logic (if (x < 10 && y != 0) {...})",
        "code": """
        int main() {
            int x = 5;
            int y = 0;
            if (x < 10 && y != 0) {
                x = x + 1;
            } else {
                x = 0;
            }
        }
        """
    },
    {
        "name": "caso_5_call_index",
        "desc": "Chamada de função, indexação e expressão (z = f(a, b)[i] * g();)",
        "code": """
        int g() { return 1; }
        int f(int a, int b) { return 1; } // Simplificado
        int main() {
            int z;
            int arr;
            int a = 1, b = 2, i = 3;
            z = f(a, b) * g();
        }
        """
    },
    {
        "name": "caso_6_erro_falta_equal",
        "desc": "ERRO: faltou '=' na atribuição (int z 7;)",
        "code": """
        int main() {
            int z 7;
        }
        """
    },
    {
        "name": "caso_7_erro_falta_rparen",
        "desc": "ERRO: parêntese aberto sem fechar (1 + (2 * 3;)",
        "code": """
        int main() {
            int result;
            result = 1 + (2 * 3;
        }
        """
    },
    {
        "name": "caso_8_erro_lonely_else",
        "desc": "ERRO: 'else' sem 'if' correspondente.",
        "code": """
        int main() {
            if (1 > 0) {}
            int x = 1;
            else { x = 0; }
        }
        """
    },
    {
        "name": "caso_9_chaves_invertidas",
        "desc": "ERRO: if com bloco de código malformado (chaves/parênteses desordenados).",
        "code": """
        int main() {
            int val = 1;
            if val > 0 {
                val = 0;
            }
        }
        """
    },
    {
        "name": "caso_10_atribuicao_sem_equal",
        "desc": "ERRO: Comando de atribuição sem o operador '='.",
        "code": """
        int main() {
            int a = 10;
            a 20;
        }
        """
    }
]


i = 1
for code in testes:
    nome = code["name"]
    code_stripped = code["code"].strip()

    diretorio = os.path.join("testes", "caso " + str(i))
    os.makedirs(diretorio, exist_ok=True)
    LOG_FILE = os.path.join(diretorio, f"{nome}.txt")
    AST_FILE = os.path.join(diretorio, f"{nome}_ast.png")
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        
        f.write(f"{code['desc']}\n")
        
        f.write("\n-- CODIGO FONTE --\n")
        f.write(code["code"] + "\n")
        
        lexer.input(code_stripped)
        f.write("\n-- TOKENS --\n")
        for t in lexer:
            f.write(f'Token(type={t.type}, value="{t.value}", line={t.lineno})\n')

        lexer.input(code_stripped)
        lexer.lineno = 1
        
        try:
            ast = parser.parse(code_stripped, lexer=lexer)
        except Exception as e:
            f.write(f"ERRO FATAL DURANTE O PARSING: {e}\n")
            ast = None
        print(nome)
        if ast and not houveErro:
            f.write("\n-- AST Gerada --\n")
            f.write(str(ast) + "\n")
            
            draw_tree(ast, AST_FILE)
            
            print("-- SUCESSO NA ANÁLISE SINTÁTICA --\n")
          
        else:
            print("-- FALHA NA ANÁLISE SINTÁTICA --\n")
    i += 1

