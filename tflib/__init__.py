import numpy as np
import tensorflow as tf
import hashlib
import os, errno, glob

import locale

locale.setlocale(locale.LC_ALL, '')

_params = {}
def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.
    
    Creates and returns theano shared variables similarly to `tf.Variable`, 
    except if you try to create a param with the same name as a 
    previously-created one, `param(...)` will just return the old one instead of 
    making a new one.

    This constructor also adds a `param` attribute to the shared variables it 
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    return _params[name]

# def search(node, critereon):
#     """
#     Traverse the Theano graph starting at `node` and return a list of all nodes
#     which match the `critereon` function. When optimizing a cost function, you 
#     can use this to get a list of all of the trainable params in the graph, like
#     so:

#     `lib.search(cost, lambda x: hasattr(x, "param"))`
#     """

#     def _search(node, critereon, visited):
#         if node in visited:
#             return []
#         visited.add(node)

#         results = []
#         if isinstance(node, T.Apply):
#             for inp in node.inputs:
#                 results += _search(inp, critereon, visited)
#         else: # Variable node
#             if critereon(node):
#                 results.append(node)
#             if node.owner is not None:
#                 results += _search(node.owner, critereon, visited)
#         return results

#     return _search(node, critereon, set())

# def print_params_info(params):
#     """Print information about the parameters in the given param set."""

#     params = sorted(params, key=lambda p: p.name)
#     values = [p.get_value(borrow=True) for p in params]
#     shapes = [p.shape for p in values]
#     print "Params for cost:"
#     for param, value, shape in zip(params, values, shapes):
#         print "\t{0} ({1})".format(
#             param.name,
#             ",".join([str(x) for x in shape])
#         )

#     total_param_count = 0
#     for shape in shapes:
#         param_count = 1
#         for dim in shape:
#             param_count *= dim
#         total_param_count += param_count
#     print "Total parameter count: {0}".format(
#         locale.format("%d", total_param_count, grouping=True)
#     )

def print_model_settings(locals_):
    print "Model settings:"
    all_vars = [(k,v) for (k,v) in locals_.items() if (k.isupper() and k!='T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)

def ensure_dir(dirname):
    """
    Ensure that a named directory exists; if it does not, attempt to create it.
    """
    try:
        os.makedirs(dirname)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise

__model_setting_file_name = 'model_settings.txt'
def print_model_settings(locals_var, path=None, sys_arg=False):
    """
    Prints all variables in upper case in locals_var,
    except for T which usually stands for theano.tensor.
    If locals() passed as input to this method, will print
    all the variables in upper case defined so far, that is
    model settings.

    With `path` as an address to a directory it will _append_ it
    as a file named `model_settings.txt` as well.

    Author: Soroush Mehri (https://github.com/soroushmehr)
    Modifier: Kundan Kumar
    """
    log = ""
    log += "\nModel settings:"
    all_vars = [(k,v) for (k,v) in locals_var.items() if (k.isupper() and k != 'T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        log += ("\n\t%-20s %s" % (var_name, var_value))

    print log

    if path is not None:
        ensure_dir(path)
        # Don't override, just append if by mistake there is something in the file.
        with open(os.path.join(path, __model_setting_file_name), 'a+') as f:
            f.write(log)
    

def get_MD5_log_hash(locals_var):
    log = "\nModel settings:"
    all_vars = [(k,v) for (k,v) in locals_var.items() if (k.isupper() and k != 'T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])

    for var_name, var_value in all_vars:
        log += ("\n\t%-20s %s" % (var_name, var_value))

    hash_object = hashlib.md5(log)
    return hash_object.hexdigest()
