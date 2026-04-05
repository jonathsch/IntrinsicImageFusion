import functools


def rsetattr(obj, attr, val):
    # using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        if attr == '':
            return obj

        if isinstance(obj, dict):
            return obj[attr]

        try:
            return getattr(obj, attr, *args)
        except AttributeError:
            # Try as a dictionary
            try:
                return obj[attr]
            except TypeError:
                # Try as a list
                return obj[int(attr)]

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):
    def _hasattr(obj, attr):
        return hasattr(obj, attr)

    return functools.reduce(_hasattr, [obj] + attr.split('.'))
