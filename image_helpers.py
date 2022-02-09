"""
Helper functions for when working with any image data

Arthur McCray
amccray@anl.gov
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import skimage
from ipywidgets import interact
from scipy import ndimage as ndi
from scipy.signal import tukey
from scipy.spatial.transform import Rotation as R


###
### Functions for displaying images
###


def show_im(
    image,
    title=None,
    simple=False,
    origin="upper",
    cbar=True,
    cbar_title="",
    scale=None,
    save=None,
    **kwargs,
):
    """Display an image on a new axis.

    Takes a 2D array and displays the image in grayscale with optional title on
    a new axis. In general it's nice to have things on their own axes, but if
    too many are open it's a good idea to close with plt.close('all').

    Args:
        image (2D array): Image to be displayed.
        title (str): (`optional`) Title of plot.
        simple (bool): (`optional`) Default output or additional labels.

            - True, will just show image.
            - False, (default) will show a colorbar with axes labels, and will adjust the
              contrast range for images with a very small range of values (<1e-12).

        origin (str): (`optional`) Control image orientation.

            - 'upper': (default) (0,0) in upper left corner, y-axis goes down.
            - 'lower': (0,0) in lower left corner, y-axis goes up.

        cbar (bool): (`optional`) Choose to display the colorbar or not. Only matters when
            simple = False.
        cbar_title (str): (`optional`) Title attached to the colorbar (indicating the
            units or significance of the values).
        scale (float): Scale of image in nm/pixel. Axis markers will be given in
            units of nanometers.

    Returns:
        None
    """
    _fig, ax = plt.subplots()
    image = np.array(image)
    if "cmap" not in kwargs:
        kwargs["cmap"] = "gray"
    if image.dtype == "bool":
        image = image.astype("int")
    if not simple and np.max(image) - np.min(image) < 1e-12:
        # adjust coontrast range
        vmin = np.min(image) - 1e-12
        vmax = np.max(image) + 1e-12
        im = ax.matshow(image, origin=origin, vmin=vmin, vmax=vmax, **kwargs)
    else:
        im = ax.matshow(image, origin=origin, **kwargs)

    if title is not None:
        ax.set_title(str(title), pad=0)

    if simple:
        plt.axis("off")
    else:
        plt.tick_params(axis="x", top=False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction="in")
        if scale is None:
            ticks_label = "pixels"
        else:

            def mjrFormatter(x):
                return f"{scale*x:.3g}"

            fov = scale * max(image.shape[0], image.shape[1])

            if fov < 4e3:  # if fov < 4um use nm scale
                ticks_label = " nm "
            elif fov > 4e6:  # if fov > 4mm use m scale
                ticks_label = "  m  "
                scale /= 1e9
            else:  # if fov between the two, use um
                ticks_label = r" $\mu$m "
                scale /= 1e3

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if origin == "lower":
            ax.text(y=0, x=0, s=ticks_label, rotation=-45, va="top", ha="right")
        elif origin == "upper":  # keep label in lower left corner
            ax.text(
                y=image.shape[0], x=0, s=ticks_label, rotation=-45, va="top", ha="right"
            )

        if cbar:
            plt.colorbar(im, ax=ax, pad=0.02, format="%.2g", label=str(cbar_title))

    if save:
        print("saving: ", save)
        plt.savefig(save, dpi=400, bbox_inches="tight")

    plt.show()
    return


def show_im_points(im=None, points=None, points2=None, size=None, title=None, **kwargs):
    """
    points an array [[y1,x1], [y2,x2], ...]
    """
    _fig, ax = plt.subplots()
    if im is not None:
        ax.matshow(im, cmap="gray", **kwargs)
    if points is not None:
        points = np.array(points)
        ax.plot(
            points[:, 1],
            points[:, 0],
            c="r",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    if points2 is not None:
        points2 = np.array(points2)
        ax.plot(
            points2[:, 1],
            points2[:, 0],
            c="b",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    ax.set_aspect(1)
    if title is not None:
        ax.set_title(str(title), pad=0)
    plt.show()
