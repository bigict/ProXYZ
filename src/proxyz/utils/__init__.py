import contextlib
import functools


class dict2object(object):
    def __init__(self, **args):
        self.__dict__.update(args)


def compose(*funcs):
  return functools.reduce(lambda g, f: lambda x: f(g(x)), funcs)


@contextlib.contextmanager
def attr(obj, **kwags):
    t = {key: getattr(obj, key) for key in kwags}

    for key in kwags:
        setattr(obj, key, kwags[key])
    yield obj
    for key in kwags:
        setattr(obj, key, t[key])
