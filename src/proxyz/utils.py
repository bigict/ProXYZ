import functools


class dict2object(object):
    def __init__(self, **args):
        self.__dict__.update(args)


def compose(*funcs):
  return functools.reduce(lambda g, f: lambda x: f(g(x)), funcs)
