""" helper functions for when working with any image data

None of these functions should take class objects, but should work on raw arrays to keep
them applicable for more situations.

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


def show_stack(images, ptie=None, origin="upper", title=False, scale_each=True):
    """Shows a stack of dm3s or np images with a slider to navigate slice axis.

    Uses ipywidgets.interact to allow user to view multiple images on the same
    axis using a slider. There is likely a better way to do this, but this was
    the first one I found that works...

    If a TIE_params object is given, only the regions corresponding to ptie.crop
    will be shown.

    Args:
        images (list): List of 2D arrays. Stack of images to be shown.
        ptie (``TIE_params`` object): Will use ptie.crop to show only the region
            that will remain after being cropped.
        origin (str): (`optional`) Control image orientation.
        title (bool): (`optional`) Try and pull a title from the signal objects.
    Returns:
        None
    """
    images = np.array(images)
    if not scale_each:
        vmin = np.min(images)
        vmax = np.max(images)

    if ptie is None:
        top, bot = 0, images[0].shape[0]
        left, r = 0, images[0].shape[1]
    else:
        if ptie.rotation != 0 or ptie.x_transl != 0 or ptie.y_transl != 0:
            rotate, x_shift, y_shift = ptie.rotation, ptie.x_transl, ptie.y_transl
            for i, _ in enumerate(images):
                images[i] = ndi.rotate(images[i], rotate, reshape=False)
                images[i] = ndi.shift(images[i], (-y_shift, x_shift))
        top = ptie.crop["top"]
        bot = ptie.crop["bottom"]
        left = ptie.crop["left"]
        r = ptie.crop["right"]

    images = images[:, top:bot, left:r]

    _fig, _ax = plt.subplots()
    plt.axis("off")
    N = images.shape[0]

    def view_image(i=0):
        if scale_each:
            _im = plt.imshow(
                images[i], cmap="gray", interpolation="nearest", origin=origin
            )
        else:
            _im = plt.imshow(
                images[i],
                cmap="gray",
                interpolation="nearest",
                origin=origin,
                vmin=vmin,
                vmax=vmax,
            )

        if title:
            plt.title("Stack[{:}]".format(i))

    interact(view_image, i=(0, N - 1))
    return


def show_fft(fft, title=None, **kwargs):
    """Display the log of the abs of a FFT

    Args:
        fft (ndarray): 2D image
        title (str, optional): title of image. Defaults to None.
        **kwargs: passed to show_im()
    """
    nonzeros = np.nonzero(fft)
    fft[nonzeros] = np.log10(np.abs(fft[nonzeros]))
    fft = fft.real
    show_im(fft, title=title, **kwargs)


def show_im_peaks(im=None, peaks=None, peaks2=None, size=None, title=None, **kwargs):
    """
    peaks an array [[y1,x1], [y2,x2], ...]
    """
    _fig, ax = plt.subplots()
    if im is not None:
        ax.matshow(im, cmap="gray", **kwargs)
    if peaks is not None:
        peaks = np.array(peaks)
        ax.plot(
            peaks[:, 1],
            peaks[:, 0],
            c="r",
            alpha=0.9,
            ms=size,
            marker="o",
            fillstyle="none",
            linestyle="none",
        )
    if peaks2 is not None:
        peaks2 = np.array(peaks2)
        ax.plot(
            peaks2[:, 1],
            peaks2[:, 0],
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


###
### Misc small helpers
###


def get_histo(im, minn=None, maxx=None, numbins=None):
    """
    gets a histogram of a list of datapoints (im), specify minimum value, maximum value,
    and number of bins
    """
    im = np.array(im)
    if minn is None:
        minn = np.min(im)
    if maxx is None:
        maxx = np.max(im)
    if numbins is None:
        numbins = min(np.size(im) // 20, 100)
        print(f"{numbins} bins")
    _fig, ax = plt.subplots()
    ax.hist(im, bins=np.linspace(minn, maxx, numbins))
    plt.show()


def get_fft(im):
    """Get fast fourier transform of 2D image"""
    return np.fft.fftshift(np.fft.fft2(im))


def get_ifft(fft):
    """Get inverse fast fourier transform of 2D image"""
    return np.fft.ifft2(np.fft.ifftshift(fft))


def Tukey2D(shape, alpha=0.5, sym=True):
    """
    makes a 2D (rectangular not round) window based on a Tukey signal
    Useful for windowing images before taking FFTs
    """
    dimy, dimx = shape
    ty = tukey(dimy, alpha=alpha, sym=sym)
    filt_y = np.tile(ty.reshape(dimy, 1), (1, dimx))
    tx = tukey(dimx, alpha=alpha, sym=sym)
    filt_x = np.tile(tx, (dimy, 1))
    output = filt_x * filt_y
    return output


def norm_image(image):
    """Normalize image intensities to between 0 and 1"""
    image = image - np.min(image)
    image = image / np.max(image)
    return image


def overwrite_rename(filepath):
    """Given a filepath, check if file exists already. If so, add numeral 1 to end,
    if already ends with a numeral increment by 1.

    Args:
        filepath (str): filepath to be checked

    Returns:
        str: [description]
    """

    def splitnum(s):
        """split the trailing number off a string. Returns (stripped_string, number)"""
        head = s.rstrip("-.0123456789")
        tail = s[len(head) :]
        return head, tail

    filepath = str(filepath)
    file, ext = os.path.splitext(filepath)
    if os.path.isfile(filepath):
        if file[-1].isnumeric():
            file, num = splitnum(file)
            nname = file + str(int(num) + 1) + ext
            return overwrite_rename(nname)
        else:
            return overwrite_rename(file + "1" + ext)
    else:
        return filepath


###
### Pretty much everything else...
###


def autocorr(arr):
    """Calculate the autocorrelation of an image
    method described in from Loudon & Midgley, Ultramicroscopy 109, (2009).

    Args:
        arr (ndarray): Image to autocorrelate

    Returns:
        ndarray: Autocorrelation
    """
    fft = get_fft(arr)
    return np.fft.ifftshift(get_ifft(fft * np.conjugate(fft))).real


def bbox(img, digits=10):
    """
    Get minimum bounding box of image, trimming off black (0) regions.
    values will be rounded to `digits` places.
    """
    img = np.round(img, digits)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return img[ymin : ymax + 1, xmin : xmax + 1]


def filter_hotpix(image, thresh=12, show=False, iters=0):
    """
    look for pixel values with an intensity >3 std outside of mean of surrounding
    8 pixels. If found, replace with median value of those pixels
    """
    # for now, return binary image 1 where filtered, 0 where not
    if iters > 10:
        print("Ended at 10 iterations of filter_hotpix.")
        return image

    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]).astype("float")
    kernel = kernel / np.sum(kernel)
    image = image.astype("float")
    mean = ndi.convolve(image, kernel, mode="reflect")
    dif = np.abs(image - mean)
    std = np.std(dif)

    bads = np.where(dif > thresh * std)
    numbads = len(bads[0])

    filtered = np.copy(image)
    filtered[bads] = mean[bads]
    if show:
        print(numbads, "hot-pixels filtered")
        show_im_peaks(
            image,
            np.transpose([bads[0], bads[1]]),
            title="hotpix identified on first pass",
        )
    if numbads > 0:
        filtered = filter_hotpix(filtered, thresh=thresh, show=False, iters=iters + 1)

    return filtered


def filter_background(
    image,
    scale=1,
    filt_hotpix=True,
    thresh=15,
    filter_lf=100,
    filter_hf=10,
    show=False,
    ret_bkg=False,
):
    """
    image: image to be filtered
    scale: scale of image in nm/pixel, this allows you to set the filter sizes in nm
    filt_hotpix: True if you want to filter hot/dead pixels, false otherwise
    thresh: threshold for hotpix filtering. Higher threshold means fewer pixels
        will be filtered
    filter_lf: low-frequency filter std in nm (pix * scale)
    filter_hf: high-frequeuency filter std in nm (pix * scale)
    ret_bkg: will return the subtracted background (no hotpix) if True
    """

    dim_y, dim_x = image.shape

    x_sampling = y_sampling = 1 / scale  # [pixels/nm]
    u_max = x_sampling / 2
    v_max = y_sampling / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, dim_x)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, dim_y)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)
    r = np.sqrt(u_mat ** 2 + v_mat ** 2)

    inverse_gauss_filter = 1 - np.exp(-1 * (r * filter_lf) ** 2)
    gauss_filter = np.exp(-1 * (r * filter_hf) ** 2)
    bp_filter = inverse_gauss_filter * gauss_filter

    if filt_hotpix:
        image = filter_hotpix(image, show=show, thresh=thresh)
    fft = get_fft(image)

    filtered_im = np.real(get_ifft(fft * inverse_gauss_filter * gauss_filter))
    dif = image - filtered_im

    if show:
        show_im(bp_filter, "filter")
        show_im(filtered_im, "filtered image", cbar=False)
        show_im(dif, "removed background", cbar=False)
        # show_fft(get_fft(dif), 'fft of background', cbar=False)

    filtered_im = norm_image(filtered_im)
    if ret_bkg:
        return (filtered_im, norm_image(dif))
    else:
        return filtered_im


def dist(ny, nx, shift=False):
    """Creates a frequency array for Fourier processing.

    Args:
        ny (int): Height of array
        nx (int): Width of array
        shift (bool): Whether to center the frequency spectrum.

            - False: (default) smallest values are at the corners.
            - True: smallest values at center of array.

    Returns:
        ``ndarray``: Numpy array of shape (ny, nx).
    """
    ly = (np.arange(ny) - ny / 2) / ny
    lx = (np.arange(nx) - nx / 2) / nx
    [X, Y] = np.meshgrid(lx, ly)
    q = np.sqrt(X ** 2 + Y ** 2)
    if not shift:
        q = np.fft.ifftshift(q)
    return q


def dist4(dim, norm=False):
    """4-fold symmetric distance map even at small radiuses

    Args:
        dim (int): desired dimension of output
        norm (bool, optional): Normalize maximum of output to 1. Defaults to False.

    Returns:
        ``ndarray``: 2D (dim, dim) array
    """
    # 4-fold symmetric distance map even at small radiuses
    d2 = dim // 2
    a = np.arange(d2)
    b = np.arange(d2)
    if norm:
        a = a / (2 * d2)
        b = b / (2 * d2)
    x, y = np.meshgrid(a, b)
    quarter = np.sqrt(x ** 2 + y ** 2)
    sym_dist = np.zeros((dim, dim))
    sym_dist[d2:, d2:] = quarter
    sym_dist[d2:, :d2] = np.fliplr(quarter)
    sym_dist[:d2, d2:] = np.flipud(quarter)
    sym_dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return sym_dist


def circ4(dim, rad):
    """Binary circle mask, 4-fold symmetric even at small dimensions"""
    return (dist4(dim) < rad).astype("int")


def lineplot_im(image, center=None, phi=0, linewidth=1, show=False, use_abs=False):
    """
    image to take line plot through
    center point (cy, cx) in pixels
    angle (deg) to take line plot, with respect to y,x axis (y points down).
        currently always does the line profile left to right,
        phi=90 will be vertical profile top to bottom
        phi = -90 will be vertical profile bottom to top
    """
    im = np.array(image)
    if np.ndim(im) > 2:
        print("More than 2 dimensions given, collapsing along first axis")
        im = np.sum(im, axis=0)
    dy, dx = im.shape
    if center is None:
        print("line through middle of image")
        center = (dy // 2, dx // 2)
    cy, cx = center[0], center[1]

    sp, ep = box_intercepts(im.shape, center, phi)
    profile = skimage.measure.profile_line(
        im, sp, ep, linewidth=linewidth, mode="constant", reduce_func=np.mean
    )
    if show:
        _fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        if use_abs:
            ax0.plot(np.abs(profile))
        else:
            ax0.plot(profile)
        ax0.set_aspect(1 / ax0.get_data_ratio(), adjustable="box")
        ax0.set_ylabel("intensity")
        ax0.set_xlabel("pixels")
        ax1.matshow(im)
        if linewidth > 1:
            th = np.arctan2((ep[0] - sp[0]), (ep[1] - sp[1]))
            spp, epp = box_intercepts(
                im.shape,
                (cy + np.cos(th) * linewidth / 2, cx - np.sin(th) * linewidth / 2),
                phi,
            )
            spm, epm = box_intercepts(
                im.shape,
                (cy - np.cos(th) * linewidth / 2, cx + np.sin(th) * linewidth / 2),
                phi,
            )
            ax1.plot([spp[1], epp[1]], [spp[0], epp[0]], color="red", linewidth=1)
            ax1.plot([spm[1], epm[1]], [spm[0], epm[0]], color="red", linewidth=1)
        else:
            ax1.plot([sp[1], ep[1]], [sp[0], ep[0]], color="red", linewidth=1)

        ax1.set_xlim([0, im.shape[1] - 1])
        ax1.set_ylim([im.shape[0] - 1, 0])

        plt.show()

    return profile


def box_intercepts(dims, center, phi):
    """
    given box of size dims=(dy, dx), a line at angle phi (deg) with respect to the x
    axis and going through point center=(cy,cx) will intercept the box at points
    sp = (spy, spx) and ep=(epy,epx) where sp is on the left half and ep on the
    right half of the box. for phi=90deg vs -90 will flip top/bottom sp ep
    """
    dy, dx = dims
    cy, cx = center
    phir = np.deg2rad(phi)
    tphi = np.tan(phir)
    tphi2 = np.tan(phir - np.pi / 2)

    # calculate the end edge
    epy = round((dx - cx) * tphi + cy)
    if 0 <= epy < dy:
        epx = dx - 1
    elif epy < 0:
        epy = 0
        epx = round(cx + cy * tphi2)
    else:
        epy = dy - 1
        epx = round(cx + (dy - cy) / tphi)

    spy = round(cy - cx * tphi)
    if 0 <= spy < dy:
        spx = 0
    elif spy >= dy:
        spy = dy - 1
        spx = round(cx - (dy - cy) * tphi2)
    else:
        spy = 0
        spx = round(cx - cy / tphi)

    sp = (spy, spx)  # start point
    ep = (epy, epx)  # end point
    return sp, ep


def total_tilt(tx, ty, xfirst=True, rad=False):
    """
    returns (altitude, azimuth) in degrees after tilting around x axis by tx
    and then y axis by ty.
    xfirst=True if rotating around x then y, affects azimuth only
    rad=False if input degrees (default) or True if input in radians
    """
    if not rad:
        tx = np.deg2rad(tx)
        ty = np.deg2rad(ty)
    Rx = R.from_rotvec(tx * np.array([1, 0, 0]))  # [x,y,z]
    Ry = R.from_rotvec(ty * np.array([0, 1, 0]))
    v = np.array([0, 0, 1])
    if xfirst:
        vrot = Ry.apply(Rx.apply(v))
    else:
        vrot = Rx.apply(Ry.apply(v))

    alt = np.arctan(np.sqrt(vrot[0] ** 2 + vrot[1] ** 2) / vrot[2])
    alt = round(np.rad2deg(alt), 13)

    az = np.rad2deg(np.arctan2(vrot[1], vrot[0]))
    return alt, az


###
### Helpers that are more specific and used by other functions I've written
### These should (and likely will) be moved out of this document at some point
###


def get_mean(pos1, pos2):
    """Mean point of two positions (2D)"""
    return ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)


def get_dist(pos1, pos2):
    """Distance between two 2D points"""
    squared = (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2
    return np.sqrt(squared)


def sort_clockwise(points, ref):
    """Sort a set of points into a clockwise order"""
    points = np.array(points)
    points = points.astype("float")
    points = np.unique(points, axis=0)
    angles = np.arctan2(points[:, 0] - ref[0], points[:, 1] - ref[1])
    ind = np.argsort(angles)
    return points[ind[::-1]]


def center_crop_square(im):
    """Crop image to square using min(dimy, dimx)"""
    dy, dx = im.shape
    if dy == dx:
        return im
    elif dy > dx:
        my = dy // 2
        dx1 = int(np.ceil(dx / 2))
        dx2 = int(np.floor(dx / 2))
        return im[my - dx1 : my + dx2, :]
    elif dy < dx:
        mx = dx // 2
        dy1 = int(np.ceil(dy / 2))
        dy2 = int(np.floor(dy / 2))
        return im[:, mx - dy1 : mx + dy2]
