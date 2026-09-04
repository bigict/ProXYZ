import os
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


def env(key, defval=None, dtype=None):
    value = os.getenv(key)
    if value is not None:
        if defval is not None and dtype is None:
            dtype = type(defval)
        if dtype == bool:
            # json-style lower-case only.
            if value.casefold() == "true":
                return True
            elif value.casefold() == "false":
                return False
            return int(value) != 0
        return dtype(value) if dtype is not None else value
    return defval
