from __future__ import annotations
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Any, Tuple, Dict
import os

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

NodeLike = Any

def node_label(n: NodeLike) -> str:
    tname = type(n).__name__
    if tname == "Program": return "Program"
    if tname == "Let":     return "Let"
    if tname == "Assign":  return "Assign"
    if tname == "If":      return "If"
    if tname == "While":   return "While"
    if tname == "Return":  return "Return"
    if tname == "Block":   return "Block"
    if tname == "Call":    return "Call"
    if tname == "Index":   return "Index"
    if tname == "BinOp":   return f"BinOp('{getattr(n, 'op', '?')}')"
    if tname == "Var":     return f"Id({getattr(n, 'name', '?')})"
    if tname == "Num":     return f"Num({getattr(n, 'value', '?')})"
    if tname == "Bool":
        v = getattr(n, "value", None)
        return f"Bool({str(v).lower()})" if isinstance(v, bool) else "Bool(?)"
    return tname

def children(n: NodeLike) -> List[NodeLike]:
    tname = type(n).__name__
    if tname == "Program": return list(getattr(n, "body", []))
    if tname == "Let":     return [getattr(n, "lhs", None), getattr(n, "init", None)]
    if tname == "Assign":  return [getattr(n, "target", None), getattr(n, "value", None)]
    if tname == "If":
        lst = [getattr(n, "test", None), getattr(n, "then", None)]
        other = getattr(n, "otherwise", None)
        if other is not None: lst.append(other)
        return lst
    if tname == "While":   return [getattr(n, "test", None), getattr(n, "body", None)]
    if tname == "Return":
        v = getattr(n, "value", None)
        return [v] if v is not None else []
    if tname == "Block":   return list(getattr(n, "body", []))
    if tname == "Call":    return [getattr(n, "callee", None)] + list(getattr(n, "args", []))
    if tname == "Index":   return [getattr(n, "target", None), getattr(n, "index", None)]
    if tname == "BinOp":   return [getattr(n, "left", None), getattr(n, "right", None)]

    if is_dataclass(n):
        out: List[Any] = []
        for k, v in n.__dict__.items():
            if k in ("line", "col", "op", "name", "value"):
                continue
            if isinstance(v, list):
                out.extend(v)
            elif v is not None:
                out.append(v)
        return out
    if hasattr(n, "children"):
        return list(getattr(n, "children"))
    return []

def _compute_layout(n: NodeLike, x0=0.0, y0=0.0, y_spacing=1.6) -> Tuple[Dict[int,Tuple[float,float]], float]:
    ch = [c for c in children(n) if c is not None]
    if not ch:
        return ({id(n): (x0, y0)}, 1.0)
    pos: Dict[int,Tuple[float,float]] = {}
    widths: List[float] = []
    subs: List[NodeLike] = []
    for c in ch:
        subpos, w = _compute_layout(c, 0, 0, y_spacing)
        pos.update(subpos)
        widths.append(w)
        subs.append(c)
    total_w = sum(widths) + (len(widths)-1)*0.8
    cur_x = x0 - total_w/2.0
    def shift(node: NodeLike, dx: float, dy: float):
        x, y = pos[id(node)]
        pos[id(node)] = (x + dx, y + dy)
        for cc in children(node):
            if cc is not None:
                shift(cc, dx, dy)
    for c, w in zip(subs, widths):
        cx = cur_x + w/2.0
        shift(c, cx, y0 - y_spacing)
        cur_x += w + 0.8
    pos[id(n)] = (x0, y0)
    return pos, total_w

def draw_tree(root: NodeLike, filename: str, figsize=(10, 7), dpi: int = 160):
    if not HAVE_MPL:
        return
    pos, _ = _compute_layout(root, 0.0, 0.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    def draw_edges(node: NodeLike):
        x, y = pos[id(node)]
        for c in children(node):
            if c is None:
                continue
            xc, yc = pos[id(c)]
            ax.plot([x, xc], [y-0.05, yc+0.05])
            draw_edges(c)
    def draw_nodes(node: NodeLike):
        x, y = pos[id(node)]
        bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
        ax.text(x, y, node_label(node), ha="center", va="center", bbox=bbox, fontsize=10)
        for c in children(node):
            if c is not None:
                draw_nodes(c)
    draw_edges(root)
    draw_nodes(root)
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    pad = 1.2
    ax.set_xlim(min(xs)-pad, max(xs)+pad)
    ax.set_ylim(min(ys)-pad, max(ys)+pad)
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

@dataclass
class Token:
    type: str
    lex: str
    line: int
    col: int

@dataclass
class Program:
    body: List[Any]

@dataclass
class Let:
    lhs: Any
    init: Any
    line: int
    col: int

@dataclass
class Assign:
    target: Any
    value: Any
    line: int
    col: int

@dataclass
class If:
    test: Any
    then: Any
    otherwise: Optional[Any]
    line: int
    col: int

@dataclass
class While:
    test: Any
    body: Any
    line: int
    col: int

@dataclass
class Return:
    value: Optional[Any]
    line: int
    col: int

@dataclass
class Block:
    body: List[Any]
    line: int
    col: int

@dataclass
class Call:
    callee: Any
    args: List[Any]
    line: int
    col: int

@dataclass
class Index:
    target: Any
    index: Any
    line: int
    col: int

@dataclass
class BinOp:
    op: str
    left: Any
    right: Any
    line: int
    col: int

@dataclass
class Var:
    name: str
    line: int
    col: int

@dataclass
class Num:
    value: str
    line: int
    col: int

@dataclass
class Bool:
    value: bool
    line: int
    col: int

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens[:]
        self.i = 0
        self.errors: List[str] = []

    def at(self):
        if self.i >= len(self.tokens):
            return Token("EOF", "", self.tokens[-1].line if self.tokens else 1, self.tokens[-1].col if self.tokens else 1)
        return self.tokens[self.i]

    def peek(self, k=0):
        idx = self.i + k
        if idx >= len(self.tokens):
            return Token("EOF", "", self.tokens[-1].line if self.tokens else 1, self.tokens[-1].col if self.tokens else 1)
        return self.tokens[idx]

    def advance(self):
        tok = self.at()
        self.i += 1
        return tok

    def match(self, types):
        if self.at().type in types:
            return self.advance()
        return None

    def token_display(self, t: Token) -> str:
        return t.lex if t.lex != "" else t.type

    def report_expected(self, expected: str, found_tok: Token):
        msg = f"Esperado {expected} (encontrado '{self.token_display(found_tok)}') @ {found_tok.line}:{found_tok.col}"
        self.errors.append(msg)

    def synchronize(self):
        sync_set = {"SEMI", "EOL", "RBRACE"}
        while self.at().type not in sync_set and self.at().type != "EOF":
            self.advance()
        if self.at().type in sync_set:
            self.advance()
        return

    def parse_program(self) -> Tuple[Program, List[str]]:
        body = []
        while self.at().type not in ("EOF", "EOL"):
            stmt = self.parse_stmt()
            if stmt:
                body.append(stmt)
            if self.match("SEMI"):
                continue
        if self.at().type == "EOL":
            self.advance()
            
        return Program(body=body), self.errors

    def parse_stmt(self):
        t = self.at()
        if t.type == "LET":
            return self.parse_let()
        elif t.type == "IF":
            return self.parse_if()
        elif t.type == "WHILE":
            return self.parse_while()
        elif t.type == "RETURN":
            return self.parse_return()
        elif t.type == "ID" and self.peek(1).type == "EQUAL":
            return self.parse_assign()
        elif t.type == "LBRACE":
            return self.parse_block()
        else:
            return self.parse_expr()

    def parse_let(self):
        let_tok = self.advance() 
        if self.at().type == "ID":
            idtok = self.advance()
            lhs = Var(name=idtok.lex, line=idtok.line, col=idtok.col)
        else:
            t = self.at()
            msg = f"Esperado identificador (encontrado '{self.token_display(t)}') @ {t.line}:{t.col}"
            self.errors.append(msg)
            self.synchronize()
            lhs = Var(name="<?>", line=t.line, col=t.col)
        if self.at().type == "EQUAL":
            self.advance()
            expr = self.parse_expr()
            return Let(lhs=lhs, init=expr, line=let_tok.line, col=let_tok.col)
        else:
            t = self.at()
            self.report_expected("'='", t)
            self.synchronize()
            if self.at().type not in ("SEMI", "EOL"):
                expr = self.parse_expr()
            else:
                expr = None
            return Let(lhs=lhs, init=expr, line=let_tok.line, col=let_tok.col)

    def parse_assign(self):
        idtok = self.advance()
        if self.at().type == "EQUAL":
            self.advance()
            expr = self.parse_expr()
            return Assign(target=Var(name=idtok.lex, line=idtok.line, col=idtok.col),
                          value=expr, line=idtok.line, col=idtok.col)
        else:
            t = self.at()
            self.report_expected("'='", t)
            self.synchronize()
            return Assign(target=Var(name=idtok.lex, line=idtok.line, col=idtok.col), value=None,
                          line=idtok.line, col=idtok.col)

    def parse_if(self):
        if_tok = self.advance() 
        #(
        if self.at().type == "LPAREN":
            self.advance()
        else:
            t = self.at()
            self.report_expected("'('", t)
            self.synchronize()
        test = self.parse_expr()
        #)
        if self.at().type == "RPAREN":
            self.advance()
        else:
            t = self.at()
            self.report_expected("')'", t)
            self.synchronize()
        then_block = self.parse_block()
        otherwise = None
        if self.at().type == "ELSE":
            self.advance()
            otherwise = self.parse_block()
        return If(test=test, then=then_block, otherwise=otherwise, line=if_tok.line, col=if_tok.col)

    def parse_while(self):
        while_tok = self.advance()
        if self.at().type == "LPAREN":
            self.advance()
        else:
            t = self.at()
            self.report_expected("'('", t)
            self.synchronize()
        test = self.parse_expr()
        if self.at().type == "RPAREN":
            self.advance()
        else:
            t = self.at()
            self.report_expected("')'", t)
            self.synchronize()
        body = self.parse_block()
        return While(test=test, body=body, line=while_tok.line, col=while_tok.col)

    def parse_return(self):
        ret_tok = self.advance()
        if self.at().type in ("SEMI", "EOL"):
            return Return(value=None, line=ret_tok.line, col=ret_tok.col)
        expr = self.parse_expr()
        return Return(value=expr, line=ret_tok.line, col=ret_tok.col)

    def parse_block(self):
        if self.at().type == "LBRACE": #{}
            l = self.advance()
            stmts = []
            while self.at().type not in ("RBRACE", "EOF"):
                stmt = self.parse_stmt()
                if stmt is not None:
                    stmts.append(stmt)
                if self.match("SEMI"):
                    continue
            if self.at().type == "RBRACE":
                self.advance()
            else:
                t = self.at()
                msg = f"Esperado '}}' (encontrado '{self.token_display(t)}') @ {t.line}:{t.col}"
                self.errors.append(msg)
                self.synchronize()
            return Block(body=stmts, line=l.line, col=l.col)
        else:
            stmt = self.parse_stmt()
            if stmt is not None:
                return Block(body=[stmt], line=stmt.line, col=stmt.col)
            else:
                return Block(body=[], line=1, col=1)

    def parse_expr(self):
        return self.parse_or()

    def parse_or(self):
        node = self.parse_and()
        while self.at().type == "OR":
            op_tok = self.advance()
            right = self.parse_and()
            node = BinOp(op="||", left=node, right=right, line=op_tok.line, col=op_tok.col)
        return node

    def parse_and(self):
        node = self.parse_equality()
        while self.at().type == "AND":
            op_tok = self.advance()
            right = self.parse_equality()
            node = BinOp(op="&&", left=node, right=right, line=op_tok.line, col=op_tok.col)
        return node

    def parse_equality(self):
        node = self.parse_relational()
        while self.at().type in ("EQ", "EQEQ", "NE", "EQUAL"):
            t = self.at()
            if t.type == "NE":
                op_tok = self.advance()
                right = self.parse_relational()
                node = BinOp(op="!=", left=node, right=right, line=op_tok.line, col=op_tok.col)
            elif t.type == "EQ" or t.type == "EQEQ" or t.type == "EQUAL":
                op_tok = self.advance()
                op = op_tok.lex if op_tok.lex in ("==", "!=") else "=="
                right = self.parse_relational()
                node = BinOp(op=op, left=node, right=right, line=op_tok.line, col=op_tok.col)
            else:
                break
        return node

    def parse_relational(self):
        node = self.parse_add()
        while self.at().type in ("LT", "LE", "GT", "GE"):
            t = self.advance()
            op = t.lex
            right = self.parse_add()
            node = BinOp(op=op, left=node, right=right, line=t.line, col=t.col)
        return node

    def parse_add(self):
        node = self.parse_mul()
        while self.at().type in ("PLUS", "MINUS"):
            t = self.advance()
            op = t.lex
            right = self.parse_mul()
            node = BinOp(op=op, left=node, right=right, line=t.line, col=t.col)
        return node

    def parse_mul(self):
        node = self.parse_postfix()
        while self.at().type in ("STAR", "SLASH"):
            t = self.advance()
            op = t.lex
            right = self.parse_postfix()
            node = BinOp(op=op, left=node, right=right, line=t.line, col=t.col)
        return node
    
    #serve tanto para operacoes binarias como numericas
    def parse_postfix(self):
        node = self.parse_primary()
        while True:
            if self.at().type == "LPAREN":
                ltok = self.advance()
                args = []
                if self.at().type != "RPAREN":
                    arg = self.parse_expr()
                    if arg:
                        args.append(arg)
                    while self.at().type == "COMMA":
                        self.advance()
                        a = self.parse_expr()
                        if a:
                            args.append(a)
                if self.at().type == "RPAREN":
                    self.advance()
                else:
                    t = self.at()
                    self.report_expected("')'", t)
                    self.synchronize()
                node = Call(callee=node, args=args, line=ltok.line, col=ltok.col)
                continue
            if self.at().type == "LBRACK":
                ltok = self.advance()
                idx = self.parse_expr()
                if self.at().type == "RBRACK":
                    self.advance()
                else:
                    t = self.at()
                    self.report_expected("']'", t)
                    self.synchronize()
                node = Index(target=node, index=idx, line=ltok.line, col=ltok.col)
                continue
            break
        return node

    def parse_primary(self):
        t = self.at()
        if t.type == "NUM":
            self.advance()
            return Num(value=t.lex, line=t.line, col=t.col)
        if t.type == "TRUE":
            self.advance()
            return Bool(value=True, line=t.line, col=t.col)
        if t.type == "FALSE":
            self.advance()
            return Bool(value=False, line=t.line, col=t.col)
        if t.type == "ID":
            self.advance()
            return Var(name=t.lex, line=t.line, col=t.col)
        if t.type == "LPAREN":
            self.advance()
            inner = self.parse_expr()
            if self.at().type == "RPAREN":
                self.advance()
            else:
                tt = self.at()
                self.report_expected("')'", tt)
                self.synchronize()
            return inner
        found = self.token_display(t)
        msg = f"Esperado número, 'true', 'false', identificador, '(' (encontrado '{found}') @ {t.line}:{t.col}"
        self.errors.append(msg)
        self.synchronize()
        return Num(value="0", line=t.line, col=t.col)

# VÁLIDOS (5)

tokens1 = [
    Token("LET","let",1,1),
    Token("ID","y",1,5),
    Token("EQUAL","=",1,7),
    Token("NUM","1",1,9),
    Token("PLUS","+",1,11),
    Token("NUM","3",1,13),
    Token("SEMI",";",1,14),
    Token("EOL","",1,15)
]
desc1 = "Atribuição simples com soma"

tokens2 = [
    Token("NUM","2",1,1),
    Token("STAR","*",1,3),
    Token("LPAREN","(",1,5),
    Token("NUM","3",1,6),
    Token("PLUS","+",1,8),
    Token("NUM","4",1,10),
    Token("RPAREN",")",1,11),
    Token("SEMI",";",1,12),
    Token("EOL","",1,13)
]
desc2 = "Expressão aritmética com parênteses"

tokens3 = [
    Token("LET","let",1,1),
    Token("ID","x",1,5),
    Token("EQUAL","=",1,7),
    Token("NUM","5",1,9),
    Token("SEMI",";",1,10),
    Token("ID","x",1,12),
    Token("SLASH","/",1,14),
    Token("NUM","2",1,16),
    Token("EOL","",1,17)
]
desc3 = "Duas instruções na mesma linha (let e depois expressão)"

tokens4 = [
    Token("IF","if",1,1),
    Token("LPAREN","(",1,3),
    Token("ID","x",1,4),
    Token("LT","<",1,6),
    Token("NUM","10",1,9),
    Token("AND","&&",1,11),
    Token("ID","y",1,14),
    Token("NE","!=",1,16),
    Token("NUM","0",1,19),
    Token("RPAREN",")",1,20),
    Token("LBRACE","{",1,22),
    Token("ID","x",1,24),
    Token("EQUAL","=",1,26),
    Token("ID","x",1,28),
    Token("PLUS","+",1,30),
    Token("NUM","1",1,32),
    Token("SEMI",";",1,33),
    Token("RBRACE","}",1,35),
    Token("ELSE","else",1,37),
    Token("ID","x",1,42),
    Token("EQUAL","=",1,44),
    Token("NUM","0",1,46),
    Token("SEMI",";",1,47),
    Token("EOL","",1,48)
]
desc4 = "if/else com && e !="

tokens5 = [
    Token("LET","let",1,1),
    Token("ID","z",1,5),
    Token("EQUAL","=",1,7),
    Token("ID","f",1,9),
    Token("LPAREN","(",1,10),
    Token("ID","a",1,11),
    Token("COMMA",",",1,12),
    Token("ID","b",1,14),
    Token("RPAREN",")",1,15),
    Token("LBRACK","[",1,16),
    Token("ID","i",1,17),
    Token("RBRACK","]",1,18),
    Token("STAR","*",1,20),
    Token("ID","g",1,22),
    Token("LPAREN","(",1,23),
    Token("RPAREN",")",1,24),
    Token("SEMI",";",1,25),
    Token("EOL","",1,26)
]
desc5 = "Chamada e indexação (sem unário)"

# INVÁLIDOS (6)

tokens6 = [
    Token("LET","let",1,1),
    Token("ID","z",1,5),
    # Token("EQUAL","=",1,7),  # ausente
    Token("NUM","7",1,9),
    Token("SEMI",";",1,10),
    Token("EOL","",1,11)
]
desc6 = "ERRO: faltou '=' na atribuição"

tokens7 = [
    Token("NUM","1",1,1),
    Token("PLUS","+",1,3),
    Token("LPAREN","(",1,5),
    Token("NUM","2",1,6),
    Token("STAR","*",1,8),
    Token("NUM","3",1,10),
    # faltou RPAREN
    Token("SEMI",";",1,11),
    Token("EOL","",1,12)
]
desc7 = "ERRO: parêntese aberto sem fechar com ')'"

tokens8 = [
    Token("ELSE","else",1,1),
    Token("ID","x",1,6),
    Token("EQUAL","=",1,8),
    Token("NUM","1",1,10),
    Token("SEMI",";",1,11),
    Token("EOL","",1,12)
]
desc8 = "ERRO: 'else' sem 'if' correspondente"

tokens9 = [
    Token("WHILE","while",1,1),
    Token("LPAREN","(",1,7),
    Token("ID","x",1,8),
    Token("LT","<",1,10),
    Token("NUM","5",1,12),
    # faltou RPAREN
    Token("LBRACE","{",1,14),
    Token("ID","x",1,16),
    Token("EQUAL","=",1,18),
    Token("ID","x",1,20),
    Token("PLUS","+",1,22),
    Token("NUM","1",1,24),
    Token("SEMI",";",1,25),
    Token("RBRACE","}",1,27),
    Token("EOL","",1,28)
]
desc9 = "ERRO: while sem fechar ')' da condição"

tokens10 = [
    Token("LET","let",1,1),
    Token("ID","a",1,5),
    Token("EQUAL","=",1,7),
    Token("ID","arr",1,9),
    Token("LBRACK","[",1,12),
    Token("NUM","1",1,13),
    Token("PLUS","+",1,15),
    Token("NUM","2",1,17),
    # faltou RBRACK
    Token("SEMI",";",1,18),
    Token("EOL","",1,19)
]
desc10 = "ERRO: indexação sem ']'"

tokens11 = [
    Token("LET","let",1,1),
    Token("ID","a",1,5),
    # Token("EQUAL","=",1,7),  # faltando de propósito
    Token("NUM","1",1,7),
    Token("SEMI",";",1,8),

    Token("ID","x",1,10),
    Token("EQUAL","=",1,12),
    # faltou expressão aqui
    Token("SEMI",";",1,13),

    Token("ID","y",1,15),
    Token("EQUAL","=",1,17),
    Token("NUM","3",1,19),
    Token("SEMI",";",1,20),
    Token("EOL","",1,21),
]


desc11 = "Dois erros, mas parser continua e aceita último stmt válido"

#let valor = 2 * (3 + 2)
tokens12 = [
    Token("LET", "let", 1,1),
    Token("ID", "valor", 1, 5),
    Token("EQUAL","=", 1, 11),
    Token("NUM","2", 1, 13),
    Token("STAR","*", 1, 15),
    Token("LPAREN","(", 1, 17),
    Token("NUM","3", 1, 18),
    Token("PLUS","+", 1, 20),
    Token("NUM","2", 1, 22),
    Token("RPAREN",")", 1, 23),
]
desc12 = "Expressão numérica"
# Runner cases
CASES = [
    ("case1_let_y",            desc1,  tokens1),
    ("case2_expr_paren",       desc2,  tokens2),
    ("case3_two_stmts",        desc3,  tokens3),
    ("case4_if_else_logic",    desc4,  tokens4),
    ("case5_call_index",       desc5,  tokens5),
    ("case6_missing_equal",    desc6,  tokens6),
    ("case7_missing_rparen",   desc7,  tokens7),
    ("case8_lonely_else",      desc8,  tokens8),
    ("case9_while_rparen",     desc9,  tokens9),
    ("case10_missing_rbrack",  desc10, tokens10),
    ("case11_two_errors_same_line", desc11, tokens11),
    ("case12_numeric_expression", desc12, tokens12),
]

def run_case(name, desc, toks):
    print(f"[CASO] {desc}")
    parser = Parser(toks)
    program, errors = parser.parse_program()
    dir = "trees/"
    if not errors:
        print("[RESULTADO] OK — AST construída")
        if HAVE_MPL:
            for id, stmt in enumerate(program.body):
                fname = f"{dir}{name}_stmt{id + 1}.png"
                draw_tree(stmt, fname)
            print(f"[ÁRVORES] salvas em 'trees/{name}_stmtNN.png'")
        else:
            print("[ÁRVORES] matplotlib não disponivel")
    else:
        print("[RESULTADO] ERROS SINTÁTICOS")
        for e in errors:
            print("  - " + e)
        print("[AST (parcial)]")
        if HAVE_MPL:
            for id, stmt in enumerate(program.body):
                fname = f"{dir}{name}_stmt{id + 1}.png"
                draw_tree(stmt, fname)
        else:
            print("[ÁRVORES] matplotlib não disponivel")

if __name__ == "__main__":
    print("=== SUÍTE DE TESTES DO PARSER ===")
    for name, desc, toks in CASES:
        print(f"\n>>> {name}")
        run_case(name, desc, toks)
        print("\n======================================================================")

