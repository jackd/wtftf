import functools
import inspect
from typing import Any, Callable

import tensorflow as tf


def export_class_method(cls: type, name: str):
    method = getattr(cls, name)

    def fn(*args, **kwargs):
        return method(*args, **kwargs)

    return fn


def as_static_method(cls: type, default_fn: Callable[[], Any], name: str):
    """
    Export f(*args, **kwargs) = `getattr(default_fn(), name)(*args, **kwargs))`.

    Exported function has same name, signature (without `self`) and docs.
    """
    method = getattr(cls, name)

    def fn(*args, **kwargs):
        return method(default_fn(), *args, **kwargs)

    sig = inspect.signature(method)
    params = tuple(sig.parameters.values())[1:]
    fn.__signature__ = sig.replace(parameters=params)
    fn.__doc__ = method.__doc__
    fn.__name__ = name
    return fn


if tf.version.VERSION < "2.4":

    def _is_symbolic_tensor(tensor):
        if isinstance(tensor, tf.Tensor):
            return hasattr(tensor, "graph")
        if isinstance(tensor, (tf.RaggedTensor, tf.SparseTensor)):
            component_tensors = tf.nest.flatten(tensor, expand_composites=True)
            return any(hasattr(t, "graph") for t in component_tensors)
        if isinstance(tensor, tf.Variable):
            return (
                getattr(tensor, "_keras_history", False) or not tf.executing_eagerly()
            )
        return False

    def _call(args, _fn, _arg_names, **kwargs):
        if tf.is_tensor(args):
            assert len(_arg_names) == 1
            args = [args]
        kwargs.update(zip(_arg_names, args))
        return _fn(**kwargs)

    def layered(fn):
        sig = inspect.signature(fn)
        arg_names = tuple(sig.parameters)

        @functools.wraps(fn)
        def ret_fn(*args, **kwargs):
            assert not any(n in kwargs for n in arg_names[: len(args)])
            kwargs.update(zip(arg_names, args))
            args = []
            names = []
            for name in arg_names:
                if name in kwargs:
                    value = kwargs[name]
                    if _is_symbolic_tensor(value):
                        args.append(value)
                        names.append(name)
            for name in names:
                del kwargs[name]

            kwargs.update(dict(_fn=fn, _arg_names=names))
            if len(args) == 0:
                return _call(args, **kwargs)

            return tf.keras.layers.Lambda(_call, arguments=kwargs)(args)

        ret_fn.__signature__ = sig
        old_doc = fn.__doc__
        ret_fn.__doc__ = (
            f"`tf.keras.layers.Lambda` wrapper around `{fn.__qualname__}`\n\n{old_doc}"
        )
        return ret_fn


else:

    def layered(fn: Callable):
        return fn
