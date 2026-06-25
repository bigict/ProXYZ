class dict2object(object):
    def __init__(self, **args):
        self.__dict__.update(args)
