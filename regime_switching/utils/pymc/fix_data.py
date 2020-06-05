import numpy as np
import pymc3 as pm
import theano


class IntData(pm.Data):
    """Hacky helper class to allow Integer data in PyMC3.
    
    See: https://github.com/pymc-devs/pymc3/issues/3493#issuecomment-638128801
    
    TODO: Remove when `pm.Data` is implemented in PyMC3 3.9
    """

    def __new__(cls, name, value):
        if isinstance(value, list):
            value = np.array(value)

        # Add data container to the named variables of the model.
        try:
            model = pm.Model.get_context()
        except TypeError:
            raise TypeError(
                "No model on context stack, which is needed "
                "to instantiate a data container. "
                "Add variable inside a 'with model:' block."
            )
        name = model.name_for(name)

        # `pm.model.pandas_to_array` takes care of parameter `value` and
        # transforms it to something digestible for pymc3
        _val = pm.model.pandas_to_array(value)
        _val = (_val).astype("int64")
        shared_object = theano.shared(_val, name)

        # To draw the node for this variable in the graphviz Digraph we need
        # its shape.
        shared_object.dshape = tuple(shared_object.shape.eval())

        model.add_random_variable(shared_object)

        return shared_object
