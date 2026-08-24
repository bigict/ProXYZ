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
