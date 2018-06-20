"""Merges lists into a single string, with the words seperated by  a space"""

def merge(list):
    l = []
    for i in list:
        for u in i:
            l.append(u)
    return " ".join(l)

list = [ 'test tes test', 'test2', 'test3', 'test4']

test = merge(list)

