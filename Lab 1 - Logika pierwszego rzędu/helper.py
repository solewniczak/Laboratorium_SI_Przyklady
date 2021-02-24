def py2pl(var):
    if type(var) is list:
        lst_string = ','.join(map(lambda item: '_' if item is None else str(item), var))
        return '[' + lst_string + ']'