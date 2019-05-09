import cProfile
import random

# Automaticly generate text from liguistic rules.

math_equation = """equation : expression symbol expression
expression : num op num | num
symbol : == | < | >
int : 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 0
num : int
op : + | - | * | /"""


def parse(description, sep=' : '):
    grammar = {}
    for line in description.split('\n'):
        des, content = line.split(sep)
        grammar[des] = [sym.split() for sym in content.split(' | ')]
    return grammar


# def gen(grammar, target):
#     '''Use bfs_search to search answer'''
#     if target not in grammar:
#         return target
#     rules = random.choice(grammar[target])
#     return " ".join(gen(grammar, r) for r in rules if r != 'null')


def gen(grammar, target):
    if target not in grammar:
        return target
    exp = random.choice(grammar[target])
    return "".join([gen(grammar, atom) for atom in exp])


def test():
    equation = gen(parse(math_equation), 'equation')
    print(equation)


cProfile.run('test()')
