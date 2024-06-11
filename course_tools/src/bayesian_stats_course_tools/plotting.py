import numpy as np

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def plot_contours(cov, pos,  nstd=1., ax=None, **kwargs):
    """
    Plot 2D parameter contours given a Hessian matrix of the likelihood
    From the jax-cosmo docs: https://jax-cosmo.readthedocs.io/en/latest/notebooks/jax-cosmo-intro.html#Where-the-wild-things-are:-Automatic-Differentiation
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    sigma_marg = lambda i: np.sqrt(cov[i, i])

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width,
                    height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    s1 = 1.5*nstd*sigma_marg(0)
    s2 = 1.5*nstd*sigma_marg(1)
    ax.set_xlim(pos[0] - s1, pos[0] + s1)
    ax.set_ylim(pos[1] - s2, pos[1] + s2)
    plt.draw()
    return ax
