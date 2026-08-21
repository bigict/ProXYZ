import contextlib
import functools


def cache(func):
    cached_func = functools.cache(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        use_cache = kwargs.pop("use_cache", True)
        if use_cache:
            return cached_func(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


class dict2object(object):
    def __init__(self, **args):
        self.__dict__.update(args)


def compose(*funcs):
    return functools.reduce(lambda g, f: lambda x: f(g(x)), funcs)


@contextlib.contextmanager
def attr(obj, **kwags):
    t = {key: getattr(obj, key) for key in kwags if hasattr(obj, key)}

    for key in kwags:
        setattr(obj, key, kwags[key])
    yield obj
    for key in kwags:
        if key in t:
            setattr(obj, key, t[key])
        else:
            delattr(obj, key)
