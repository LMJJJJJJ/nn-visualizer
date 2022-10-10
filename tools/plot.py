import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from .trajectory import smoothing_trajectory_B_spline


def plot_trajectory(ax, points, cmap, n_interpolate=500):
    dim = points.shape[-1]
    trajectory_ = smoothing_trajectory_B_spline(points, n_interpolate)
    trajectory_ = trajectory_.reshape(-1, 1, dim)
    segments = np.concatenate([trajectory_[:-1], trajectory_[1:]], axis=1)
    cols = np.linspace(0, 1, n_interpolate)
    norm = plt.Normalize(cols.min(), cols.max())
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, joinstyle="round", capstyle="round", alpha=0.05)
    lc.set_array(cols)
    lc.set_linewidth(2)
    ax.add_collection(lc)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)