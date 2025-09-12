from utils import *

i = 0
lista_tokens = []

def incrementar(passos = 1):
    global i 
    i += passos

def proximo_char(codigo):
    prox_index = i + 1
    if prox_index < len(codigo):
        return codigo[prox_index]
    return ""

def add_lista_tokens(tipo, lex, atributo=None):
    lista_tokens.append({"tipo": tipo, "lexema": lex, "atributo": atributo})

def analisador_lexico(codigo):
    
    tab_simbolos = {}
    caracteres_pulaveis = " \t\n\r"  
    id_simbolo = 1

    while i < len(codigo):
        caractere_atual = codigo[i]

        #pular caracteres como espaco, tabulacao, e paragrafo alem da quebra de linha
        if caractere_atual in caracteres_pulaveis:
            incrementar()
            continue

        #comentario singular
        if caractere_atual == "/" and proximo_char(codigo) == "/":
            while i < len(codigo) and codigo[i] != "\n":
                incrementar()
            continue

        #comentario em bloco
        if caractere_atual == "/" and proximo_char(codigo) == "*":
            lex = caractere_atual + proximo_char(codigo)
            incrementar(2)
            while i < len(codigo) - 1 and not (codigo[i] == "*" and proximo_char(codigo) == "/"):
                incrementar()
                
                if i == len(codigo) - 2 and not (codigo[i] == "*" and proximo_char(codigo) == "/"):
                    add_lista_tokens("ERRO", lex)
                    
            incrementar(2)
            continue

        #pp_directive #include, etc
        if caractere_atual == "#":
            lex = caractere_atual
            incrementar()
            while i < len(codigo) and codigo[i] != "\n":
                lex += codigo[i]
                incrementar()
            add_lista_tokens("PP_DIRECTIVE", lex)
            continue

        #tratar palavras (keywords ou nomes)
        if caractere_atual.isalpha() or caractere_atual == "_":
            lex = caractere_atual
            incrementar()
            while i < len(codigo) and ((str(codigo[i]).isalpha() or codigo[i] == "_") or str(codigo[i]).isdigit()):
                lex += codigo[i]
                incrementar()
            if lex in KEYWORDS:
                add_lista_tokens("KEYWORD", lex)
            else:
                id = "id"+ str(id_simbolo) 
                add_lista_tokens("IDENTIFICADOR", lex, id)

                if lex not in tab_simbolos.keys():
                    tab_simbolos[lex] = []
                tab_simbolos[lex].append(id)

                id_simbolo +=1
            continue

        #para numeros
        if caractere_atual.isdigit():
            lex = caractere_atual
            incrementar()
            is_float = False
            while i < len(codigo) and (codigo[i].isdigit() or codigo[i] == "." or codigo[i].isalpha()):
                if codigo[i] == ".":
                    if is_float:
                        add_lista_tokens("ERRO", lex + ".")
                        break
                    is_float = True
                #detectar erro em variavel iniciando com numero
                elif codigo[i].isalpha():
                    while i < len(codigo) and codigo[i].isalnum():
                        lex += codigo[i]
                        incrementar()
                    add_lista_tokens("ERRO", lex)
                    break
                lex += codigo[i]
                incrementar()
            else:
                if "," in lex:
                    add_lista_tokens("ERRO", lex)
                elif is_float:
                    add_lista_tokens("FLOAT", lex, "FLOAT")
                else:
                    add_lista_tokens("INTEGER", lex, "INT")
            continue

        #para string
        if caractere_atual == '"':
            lex = caractere_atual
            incrementar()
            while i < len(codigo) and codigo[i] != '"':
                lex += codigo[i]
                incrementar()
            if i < len(codigo) and codigo[i] == '"':
                lex += '"'
                add_lista_tokens("STRING", lex, "STR")
                incrementar()
            else:
                add_lista_tokens("ERRO", lex)
            continue

        #para caractere
        if caractere_atual == "'":
            lex = caractere_atual
            incrementar()
            if i < len(codigo) and codigo[i] != "'":
                lex += codigo[i]
                incrementar()
            if i < len(codigo) and codigo[i] == "'":
                lex += "'"
                add_lista_tokens("CHAR", lex, "CHAR")
                incrementar()
            else:
                add_lista_tokens("ERRO", lex)
            continue

        
        duplo_caractere = caractere_atual + proximo_char(codigo)
        if duplo_caractere in OPERADORES or duplo_caractere in DELIMITADORES:
            add_lista_tokens("OPERADOR" if duplo_caractere in OPERADORES else "DELIMITADOR", duplo_caractere)
            incrementar(2)
            continue
        elif caractere_atual in OPERADORES:
            add_lista_tokens("OPERADOR", caractere_atual)
            incrementar()
            continue
        elif caractere_atual in DELIMITADORES:
            add_lista_tokens("DELIMITADOR", caractere_atual)
            incrementar()
            continue

        #outros caracteres aleatorios
        add_lista_tokens("ERRO", caractere_atual)
        incrementar()

    return lista_tokens, tab_simbolos

testes = ['''
#include <stdio.h>
int main() {
    float p_i = 3.14;
    9pi
    char ch = 'a;
    char ch1 = "aaaa";
    char newline = 'b';newline
    int count = 0;
    // Comentário de linha
    /* Comentário de bloco 
    ainda é comentario aqui
    aqui tb*/
    *
    if (count < 10) {
        count += 1;
    }
    return 0;
}
''',
'''
#include <stdio.h>
int main(void) {
printf("Hello, world!\n");

return 0;
}
''',
'''
int main(void) {
int x = 42;
float y = 3.14;
char c = 'a';
x = x + 10;
y = y * 2
c = '\n';
return x;
}
''',
'''
int main(void) {
int i = 0;
while (i < 5) {
if (i % 2 == 0) {
printf("even\n");
} else {
printf("odd\n");
}
i++;
}
return 0;
}
''']

tokens, tab_simbolos = analisador_lexico(testes[3])
print("\nTABELA DE TOKENS:", end="\n\n")
print(f"{'Tipo':<30} {'Tokens':<30} {'Atributos'}")
print("-" * 80)
for x in tokens:

    print(f"{x['tipo']:<30} {repr(x['lexema']):<30} {x['atributo'] if x['atributo'] else ''}")

print("\nTABELA DE SIMBOLOS:", end="\n\n")

print(f"{'IDs':<30} {'Simbolos':<30} {'Quantidade'}")
print("-" * 80)

for key, val in tab_simbolos.items():
    ids = ", ".join(val)
    print(f"{ids:<30} {key:<30} {len(val)}")
