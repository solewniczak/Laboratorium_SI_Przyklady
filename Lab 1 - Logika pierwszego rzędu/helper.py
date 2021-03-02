def py2pl(var):
    if type(var) is list:
        lst_string = ','.join(map(py2pl, var))
        return '[' + lst_string + ']'
    elif var is None:
        return '_'
    else:
        return str(var)