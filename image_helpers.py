import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage as ndi
import skimage
from ipywidgets import interact



"""
helper functions for when working with any image data

None of these functions should take class objects (like Tilt_im or Skyrm or State)
but should work on raw arrays to keep them wide reaching
those types of functions can go in skyrm_helpers or similar files


"""


def show_im(image, title=None, simple=False, origin='upper', cbar=True,
    cbar_title='', scale=None, save=None, **kwargs):
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
    fig, ax = plt.subplots()
    image = np.array(image)
    if image.dtype == 'bool':
        image = image.astype('int')
    if not simple and np.max(image) - np.min(image) < 1e-12:
        # adjust coontrast range
        vmin = np.min(image) - 1e-12
        vmax = np.max(image) + 1e-12
        im = ax.matshow(image, cmap='gray', origin=origin, vmin=vmin, vmax=vmax)
    else:
        im = ax.matshow(image, cmap='gray', origin=origin, **kwargs)

    if title is not None: 
        ax.set_title(str(title), pad=0)

    if simple:
        plt.axis('off')
    else:
        plt.tick_params(axis='x',top=False)
        ax.xaxis.tick_bottom()
        ax.tick_params(direction='in')
        if scale is None:
            ticks_label = 'pixels'
        else:
            def mjrFormatter(x, pos):
                return f"{scale*x:.3g}"

            fov = scale * max(image.shape[0], image.shape[1])

            if fov < 4e3: # if fov < 4um use nm scale
                ticks_label = ' nm '
            elif fov > 4e6: # if fov > 4mm use m scale
                ticks_label = "  m  "
                scale /= 1e9
            else: # if fov between the two, use um
                ticks_label = " $\mu$m "
                scale /= 1e3

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(mjrFormatter))

        if origin == 'lower': 
            ax.text(y=0,x=0,s=ticks_label, rotation=-45, va='top', ha='right')
        elif origin =='upper': # keep label in lower left corner
            ax.text(y=image.shape[0],x=0,s=ticks_label, rotation=-45, va='top', ha='right')

        if cbar: 
            plt.colorbar(im, ax=ax, pad=0.02, format="%.2g", label=str(cbar_title))

    if save:
        print('saving: ', save)
        plt.savefig(save, dpi=400,bbox_inches='tight')

    plt.show()
    return



def show_stack(images, ptie=None, origin='upper', title=False, scale_each=True):
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
        t , b = 0, images[0].shape[0]
        l , r = 0, images[0].shape[1]
    else:
        if ptie.rotation != 0 or ptie.x_transl != 0 or ptie.y_transl != 0:
            rotate, x_shift, y_shift = ptie.rotation, ptie.x_transl, ptie.y_transl
            for i in range(len(images)):
                images[i] = ndimage.rotate(images[i], rotate, reshape=False)
                images[i] = ndimage.shift(images[i], (-y_shift, x_shift))
        t = ptie.crop['top']
        b = ptie.crop['bottom']
        l = ptie.crop['left']
        r = ptie.crop['right']

    images = images[:,t:b,l:r]

    fig, ax = plt.subplots()
    plt.axis('off')
    N = images.shape[0]

    def view_image(i=0):
        if scale_each: 
            im = plt.imshow(images[i], cmap='gray', interpolation='nearest', origin=origin)
        else:
            im = plt.imshow(images[i], cmap='gray', interpolation='nearest',
             origin=origin, vmin=vmin, vmax=vmax)

        if title: 
            plt.title('Stack[{:}]'.format(i))

    interact(view_image, i=(0, N-1))
    return 


def get_histo(im, minn=None, maxx=None, numbins=None):
    '''
    gets a histogram of a list of datapoints (im), specify minimum value, maximum value, and number of bins
    '''
    im = np.array(im)
    if minn is None:
        minn = np.min(im)
    if maxx is None:
        maxx = np.max(im)
    if numbins is None: 
        numbins = min(np.size(im)//20, 100)
        print(f"{numbins} bins")
    fig,ax = plt.subplots()
    ax.hist(im,bins=np.linspace(minn,maxx,numbins))   
    plt.show() 
    

def get_fft(im):
    return np.fft.fftshift(np.fft.fft2(im))


def get_ifft(fft):
    return np.fft.ifft2(np.fft.ifftshift(fft))


def show_fft(fft, title=None, **kwargs):
    nonzeros = np.nonzero(fft)
    fft[nonzeros] = np.log10(np.abs(fft[nonzeros]))
    fft = fft.real
    show_im(fft, title=title, **kwargs)


def show_im_peaks(im=None, peaks=None, peaks2=None, size=None, title=None, **kwargs):
    """
    peaks an array [[y1,x1], [y2,x2], ...]
    """
    fig, ax = plt.subplots()
    if im is not None: 
        ax.matshow(im, cmap='gray', **kwargs)
    if peaks is not None: 
        peaks = np.array(peaks)
        ax.plot(peaks[:,1], peaks[:,0], c='r', alpha=0.9, ms=size,
                marker='o', fillstyle='none', linestyle='none')
    if peaks2 is not None: 
        peaks2 = np.array(peaks2)
        ax.plot(peaks2[:,1], peaks2[:,0], c='b', alpha=0.9, ms=size,
                marker='o', fillstyle='none', linestyle='none')
    ax.set_aspect(1)
    if title is not None: 
        ax.set_title(str(title), pad=0)
    plt.show()
    

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
    return img[ymin:ymax+1, xmin:xmax+1]

def norm_image(image):
    image = image-np.min(image)
    image = image/np.max(image)
    return image



def filter_hotpix(image, thresh=12, show=False):
    """ 
    look for pixel values with an intensity >3 std outside of mean of surrounding
    8 pixels. If found, replace with median value of those pixels 
    """    

    # for now, return binary image 1 where filtered, 0 where not 
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]).astype('float')
    kernel = kernel / np.sum(kernel)
    image = image.astype('float')
    mean = ndi.convolve(image, kernel, mode='reflect')
    dif = np.abs(image-mean)
    std = np.std(dif)
    
    bads = np.where(dif>thresh*std)
    numbads = len(bads[0])
    
    filtered = np.copy(image)
    filtered[bads] = mean[bads]
    if show: 
        print(numbads, 'hot-pixels filtered')
        show_im_peaks(image, np.transpose([bads[0],bads[1]]), title='hotpix identified')
        # show_im_peaks(filtered, np.transpose([bads[0],bads[1]]))
    if numbads > 0: 
        filtered = filter_hotpix(filtered, thresh=thresh, show=False)
    return filtered


def local_meanstd(image): 
    """
    calculate mean and standard deviation of the surrounding 8 
    pixels for each pixel in an image. 
    From Var(X) = E(X^2) - (E(X))^2
    """
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]]).astype('float')
    kernel = kernel / np.sum(kernel)
    image = image.astype('float')
    mean = ndi.convolve(image, kernel, mode='reflect')
    im2 = image**2
    Ex2 = ndi.convolve(im2, kernel, mode='reflect')
    Ex_2 = mean**2
    std = np.sqrt(Ex2 - Ex_2)
    return mean, std


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
    ly = (np.arange(ny)-ny/2)/ny
    lx = (np.arange(nx)-nx/2)/nx
    [X,Y] = np.meshgrid(lx, ly)
    q = np.sqrt(X**2 + Y**2)
    if not shift:
        q = np.fft.ifftshift(q)
    return q



def dist4(dim, norm=False): 
    # 4-fold symmetric distance map even at small radiuses
    d2 = dim//2
    a = np.arange(d2)
    b = np.arange(d2)
    if norm:
        a = a/(2*d2)
        b = b/(2*d2)
    x,y = np.meshgrid(a,b)
    quarter = np.sqrt(x**2 + y**2)
    dist = np.zeros((dim, dim))
    dist[d2:, d2:] = quarter
    dist[d2:, :d2] = np.fliplr(quarter)
    dist[:d2, d2:] = np.flipud(quarter)
    dist[:d2, :d2] = np.flipud(np.fliplr(quarter))
    return dist


def circ4(dim, rad): 
    # 4-fold symmetric circle even at small dimensions
    return (dist4(dim)<rad).astype('int')


def lineplot_im(image, center=None, phi=0, linewidth=1, show=False, abs=False): 
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
        print('line through middle of image')
        center = (dy//2, dx//2)
    cy, cx = center[0], center[1]

    sp, ep = box_intercepts(im.shape, center, phi)
    profile = skimage.measure.profile_line(im, sp, ep, linewidth=linewidth,
                                          mode='constant', reduce_func=np.mean)
    if show: 
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        if abs:
            ax0.plot(np.abs(profile))
        else:
            ax0.plot(profile)
        ax0.set_aspect(1/ax0.get_data_ratio(), adjustable='box')
        ax0.set_ylabel('intensity')
        ax0.set_xlabel('pixels')
        ax1.matshow(im)
        if linewidth > 1: 
            th = np.arctan2((ep[0]-sp[0]), (ep[1]-sp[1]))
            spp, epp = box_intercepts(im.shape, (cy+np.cos(th)*linewidth/2, cx-np.sin(th)*linewidth/2), phi)
            spm, epm = box_intercepts(im.shape, (cy-np.cos(th)*linewidth/2, cx+np.sin(th)*linewidth/2), phi)
            ax1.plot([spp[1], epp[1]], [spp[0], epp[0]], color='red', linewidth=1)
            ax1.plot([spm[1], epm[1]], [spm[0], epm[0]], color='red', linewidth=1)
        else:
            ax1.plot([sp[1], ep[1]], [sp[0], ep[0]], color='red', linewidth=1)

        ax1.set_xlim([0,im.shape[1]-1])
        ax1.set_ylim([im.shape[0]-1, 0])

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
    tphi2 = np.tan(phir-np.pi/2)

    # calculate the end edge
    epy = round((dx-cx)*tphi + cy)
    if 0 <= epy < dy:
        epx = dx-1
    elif epy < 0: 
        epy = 0
        epx = round(cx + cy*tphi2)
    else:       
        epy = dy-1
        epx = round(cx + (dy-cy)/tphi)
    
    spy = round(cy - cx*tphi)
    if 0 <= spy < dy: 
        spx = 0
    elif spy >= dy: 
        spy = dy-1
        spx = round(cx - (dy-cy)*tphi2)
    else:
        spy = 0
        spx = round(cx - cy/tphi)
        
    sp = (spy, spx) # start point
    ep = (epy, epx) # end point
    return sp, ep


#### Helpers that are more specific to the psi6 implementation 

def get_mean(pos1, pos2):
    return ((pos1[0]+pos2[0])/2,(pos1[1]+pos2[1])/2)


def get_dist(pos1, pos2):
    squared = (pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2
    return np.sqrt(squared)

def sort_clockwise(points, ref):
    points = np.array(points)
    points = points.astype('float') # not sure why it wouldn't, but happened once
    points = np.unique(points, axis=0)
    angles = np.arctan2(points[:,0]-ref[0], points[:,1]-ref[1])
    ind = np.argsort(angles)
    return points[ind[::-1]]