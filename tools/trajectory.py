import numpy as np
import scipy.interpolate as interp

def smoothing_trajectory_B_spline(trajectory, num_points, degree=3):
    '''

    :param trajectory: [N, d]
    :param num_points:
    :param degree:
    :return:
    '''
    # delete duplicates first
    duplicates = []
    for i in range(1, len(trajectory)):
        if np.allclose(trajectory[i], trajectory[i-1]):
            duplicates.append(i)
    if duplicates:
        trajectory = np.delete(trajectory, duplicates, axis=0)
    tck, u = interp.splprep(trajectory.T, s=0, k=degree)
    u = np.linspace(0.0, 1.0, num_points)
    return np.column_stack(interp.splev(u, tck))