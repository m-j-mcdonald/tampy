from IPython import embed as shell
import numpy as np

class Matrix(np.ndarray):
    """
    The matrix class is useful for tracking object poses.
    """
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError("Override this.")

class Vector2d(Matrix):
    """
    The NAMO domain uses the Vector2d class to track poses of objects in the grid.
    """
    def __new__(cls, vec):
        if type(vec) is str:
            if not vec.endswith(")"):
                vec += ")"
            vec = eval(vec)
        obj = np.array(vec)
        assert len(obj) == 2
        obj = obj.reshape((2, 1))
        return obj
