import numpy as np
import math

class SphericalRotation3d(object):

    def __init__(self, elev, azim, degree=True):
        '''
             first R_z(azim), then R_y(elev)
        '''
        if not degree: raise Exception(f"only input in degrees is supported")
        sin = np.sin
        cos = np.cos
        deg2rad = np.deg2rad
        elev = deg2rad(elev)
        azim = deg2rad(azim)
        Rz = np.array([
            [cos(azim), -sin(azim), 0],
            [sin(azim),  cos(azim), 0],
            [    0,         0,      1]
        ])
        Ry = np.array([
            [ cos(elev), 0, sin(elev)],
            [     0,     1,      0   ],
            [-sin(elev), 0, cos(elev)]
        ])
        self.rot_mat = np.matmul(Ry, Rz)

    def __call__(self, x):
        '''
        perform the rotation
        :param x: [..., 3] np.ndarray
        :return:
        '''
        return np.matmul(x, self.rot_mat.T)


def cart2sph(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    azim = math.atan2(y, x) * 180 / math.pi
    elev = math.acos(z / r) * 180 / math.pi
    return r, azim, elev