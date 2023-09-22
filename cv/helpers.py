import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import skimage as sk
import skimage.exposure as ske
import skimage.io as skio


def imopen(fname, DATA_DIR="", as_float=True, **kwargs):
    fin = skio.imread(DATA_DIR + fname, **kwargs)
    return sk.img_as_float(fin) if as_float else fin


def show(im, axis=False, figsize=None, title=None, call_show=False, **kwargs) -> None:
    if figsize:
        plt.figure(figsize=figsize)
    if im.ndim == 2:
        kwargs.setdefault("cmap", "gray")
    if not axis:
        plt.axis("off")
        plt.tight_layout()
    plt.imshow(im, **kwargs)  # skio.imshow(im, **kwargs)
    if title:
        plt.title(title)
    if call_show:
        plt.show()


def plot3D(im, title=None, res=1):
    nx, ny = im.shape
    nx, ny = nx * res, ny * res
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    im = sk.transform.resize(im, (nx, ny))
    im = ax.plot_surface(xv, yv, im)
    if title:
        plt.title(title)


def save(fname, OUT_DIR="", ftype=".png", **kwargs):
    def rmext(fname):
        return ".".join(fname.split(".")[:-1])

    fname = rmext(fname) + ftype

    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("transparent", True)
    kwargs.setdefault("pad_inches", 0)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR + fname, **kwargs)


def rescale(im, new_width=None, new_height=None, **kwargs):
    assert new_width or new_height
    h, w = im.shape[:2]
    scale = new_width / w if new_width else new_height / h
    new_width, new_height = int(w * scale), int(h * scale)
    kwargs.setdefault("anti_aliasing", True)
    return sk.transform.resize(im, (new_height, new_width), **kwargs)


def normalize_vals(im, lo=0, hi=1):
    if not isinstance(im, np.ndarray):
        im = np.asarray(im)
    im = im - np.min(im)
    im = im * (hi - lo) / np.max(im) + lo
    return im


def normalize_cdf(im):
    """Normalize the image histogram to uniform (linear CDF)"""
    img_cdf, bin_centers = ske.cumulative_distribution(im)
    return np.interp(im, bin_centers, img_cdf)


def getGaussian2D(ksize=12, sigma=3, norm=True):
    # oneD = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    # twoD = oneD @ oneD.T
    twoD = (oneD := cv2.getGaussianKernel(ksize=ksize, sigma=sigma)) @ oneD.T
    return twoD / np.sum(twoD) if norm else twoD


def convolve2d(im, kernel, **kwargs):
    kwargs.setdefault("boundary", "symm")
    kwargs.setdefault("mode", "same")
    if im.ndim == 3 and im.shape[-1] != 1:
        return np.dstack(
            [ss.convolve2d(im[:, :, i], kernel, **kwargs) for i in range(3)]
        )
    else:
        return ss.convolve2d(im, kernel, **kwargs)
