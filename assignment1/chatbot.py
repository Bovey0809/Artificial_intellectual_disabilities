#########################################
# Created on June 02, 2019 16:22:47 JST #
#                                       #
# @author: HOU BOWEI                    #
#                                       #
# Mail:  my364007886@gmail.com          #
#########################################

'''
DOCDESIGN
TODO: 修改代码变成汉语的版本
TODO: 给每一个function写上comment
'''


def print_aug(f):
    def _f(*args):
        print(args)
        return f(args)
    return _f


def is_variable(pat: str) -> bool:
    '''
    Find *variable* in the input.
    variable is a string start with ?

    DOCTEST
    =======================
    >>> is_variable('?')
    True
    >>> is_variable('?1234')
    False
    >>> is_variable("?test")
    True
    '''
    return pat.startswith('?') and all(s.isalpha() for s in pat[1:])


def pat_match(pattern: str, saying: str) -> tuple:
    '''
    Compare saying with pattern and find the *variable*s.
    And the only difference should be the *variable*.
    Return the *varible*s pairs.

    DOCTEST
    ========================================================================
    >>> pat_match('I want ?X'.split(), "I want holiday".split())
    [('?X', 'holiday')]
    >>> pat_match('I have dreamed a ?X'.split(), "I dreamed about dog".split())
    []
    >>> pat_match('I dreamed about ?X'.split(), "I dreamed about dog".split())
    [('?X', 'dog')]
    >>> pat_match('I dreamed about ?X and a ?Y'.split(), "I dreamed about dog and a cat".split())
    [('?X', 'dog'), ('?Y', 'cat')]
    '''
    if not pattern or not saying:
        return []
    if is_variable(pattern[0]):
        return [(pattern[0], saying[0])] + pat_match(pattern[1:], saying[1:])
    else:
        if pattern[0] != saying[0]:
            return []
        else:
            return pat_match(pattern[1:], saying[1:])


def pat_to_dict(patterns):
    '''True the paird patterns into dictionary'''
    return {k: v for k, v in patterns}


@print_aug
def substitute(rule: str, parsed_rules: dict) -> list:
    '''
    >>> got_patterns = pat_match("I want ?X", "I want Iphone")
    >>> substitute("What if you mean if you got a ?X".split(), pat_to_dict(got_patterns))
    ['What', 'if', 'you', 'mean', 'if', 'you', 'got', 'a', 'Iphone']
    '''
    if not rule:
        return []
    return [parsed_rules.get(rule[0], rule[0])] + substitute(rule[1:], parsed_rules)




