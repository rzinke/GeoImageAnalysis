'''
SHORT DESCRIPTION
Viewing functions.

FUTUTRE IMPROVEMENTS
    * raster_multiplot

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltColors
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from osgeo import gdal

from ColorBalancing import equalize_image
from GeoFormatting import DS_to_extent


### STATISTICS ---
def image_stats(img, mask=None, verbose=False):
    ''' Compute the essential statistics of an image. '''
    # Setup
    stats = {}  # empty dictionary
    stats['Yshape'], stats['Xshape'] = img.shape

    # Mask image if mask is provided
    if mask is not None:
        img = np.ma.array(img, mask=(mask==0))
        img = img.compressed().flatten()

    # Compute statistics
    stats['mean'] = np.mean(img)
    stats['median'] = np.median(img)
    stats['min'] = np.min(img)
    stats['max'] = np.max(img)

    # Report if requested
    if verbose == True:
        print('''Image statistics
    shape: {Yshape:d} x {Xshape:d}
    mean: {mean:.3e}
    median: {median:.3e}
    min: {min:.3e}
    max: {max:.3e}'''.format(**stats))

    return stats


def image_percentiles(img, minPct=1, maxPct=99, verbose=False):
    ''' Find vmin and vmax for an image based on percentiles. '''
    # Determine if masked array
    if type(img) in [np.ma.array, np.ma.core.MaskedArray]:
        img = img.compressed()

    # Flatten to 1D array
    img = img.flatten()

    # Compute percentiles
    vmin, vmax = np.percentile(img, (minPct, maxPct))

    # Report if requested
    if verbose == True: print('Clipping image to {:.1f} and {:.1f} percentiles'.format(minPct, maxPct))

    return vmin, vmax


def image_clip_values(img, vmin, vmax, minPct, maxPct, mask=None, verbose=False):
    '''
    Determine vmin and vmax for an image given clip values as percentiles or
     values.
    '''
    # Apply mask if provided
    if mask is not None:
        img = np.ma.array(img, mask=(mask==0))

    # Determine clip values
    if minPct is not None or maxPct is not None:
        minClip, maxClip = image_percentiles(img, minPct, maxPct)

    # Set clip values
    if vmin is None and minPct is not None:
        vmin = minClip  # set min clip value
    if vmax is None and maxPct is not None:
        vmax = maxClip  # set max clip value

    # Report if requested
    if verbose == True:
        print('Clipping image to {:f} and {:f} '.format(vmin, vmax))

    return vmin, vmax


def dataset_clip_values(imgs, minPct=0, maxPct=100, masks=None, bounds='outer',
        verbose=False):
    '''
    Determine the vmin and vmax for a set of images provided as a list or dict.
    'Outer' bounds gives min(min)/max(max); 'inner' bounds give max(min)/min(max)
    '''
    # Convert dictionary to list
    if type(imgs) == dict:
        imgs = list(imgs.values())

    assert type(imgs) == list, 'Provide images as list or dict'

    # Empty lists of min/max values
    imgMins = []; imgMaxs = []

    # Handle masking
    if masks is None:
        masks = [None]*len(imgs)

    # Loop through images
    for i, img in enumerate(imgs):
        # Determine image clip values
        imgMin, imgMax = image_clip_values(img, vmin=None, vmax=None,
            minPct=minPct, maxPct=maxPct, mask=masks[i])

        # Append clip values to lists
        imgMins.append(imgMin)
        imgMaxs.append(imgMax)

    # Determine overall min/max
    if bounds == 'outer':
        imgMin = np.min(imgMins)
        imgMax = np.max(imgMaxs)
    elif bounds == 'inner':
        imgMin = np.max(imgMins)
        imgMax = np.min(imgMins)

    # Report if requested
    if verbose == True:
        print('Data set min: {:f}\nData set max: {:f}'.format(imgMin, imgMax))

    return imgMin, imgMax


def plot_histogram_stem(imgValues, nBins, fig, ax):
    ''' Plot the histogram generated by image_histogram as a stem plot. '''
    # Compute histogram
    hvals, hedges = np.histogram(imgValues, bins=nBins)
    hcenters = (hedges[:-1]+hedges[1:])/2

    # Plot histogram
    markerline, stemlines, baseline = ax.stem(hcenters, hvals,
        linefmt='r', markerfmt='', use_line_collection=True)
    stemlines.set_linewidths(None)
    baseline.set_linewidth(0)
    ax.plot(hcenters, hvals, 'k', linewidth=2)

    return fig, ax

def plot_histogram_kde(imgValues, nBins, fig, ax):
    ''' Plot the histogram generated by image_histogram as a smooth plot. '''
    # Downsample image
    maxValues = 1E4
    skips = np.ceil(len(imgValues)/maxValues).astype(int)
    dsampImg = imgValues[::skips]

    # Range of values
    valueRng = np.linspace(dsampImg.min(), dsampImg.max(), nBins)

    # Create histogram of image values
    kde = gaussian_kde(dsampImg)
    hist = kde(valueRng)

    # Plot interpolated values
    ax.plot(valueRng, hist, color='b', linewidth=2, label='orig values')

    return fig, ax

def plot_histogram(img, mask=None, nBins=128, plotType='stem', fig=None, ax=None):
    '''
    Plot a histogram of image values in the form of a stem plot or a
     smoothed KDE approximation.
    '''
    # Mask image
    if mask is not None:
        # Apply mask
        imgValues = np.ma.array(img, mask=(mask==0)).compressed().flatten()
    else:
        # No mask - use all values
        imgValues = img.flatten()

    # Spawn new figure and axis if necessary
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Plot histogram by specified type
    if plotType == 'stem':
        plot_histogram_stem(imgValues, nBins, fig, ax)
    elif plotType == 'kde':
        plot_histogram_kde(imgValues, nBins, fig, ax)
    else:
        print('Histogram type not recognized. Use \'stem\' or \'kde\'')

    # Format histogram
    ax.set_yticks([])

    return fig, ax




### RASTERS ---

# >>>>>>
def plot_raster(img, mask=None, extent=None,
        cmap='viridis', cbarOrient=None,
        vmin=None, vmax=None, minPct=None, maxPct=None,
        equalize=False,
        fig=None, ax=None):
    ''' Basic function to plot a single raster image. '''
    # Determine if provided image is a GDAL dataset
    if type(img) == gdal.Dataset:
        # Determine extent unless previously specified
        if extent is None:
            extent = DS_to_extent(img)

        # Retrieve image
        img = img.GetRasterBand(1).ReadAsArray()

    # Replace NaNs with zeros
    img[np.isnan(img) == 1] = 0

    # Equalize if requested
    if equalize == True:
        img = equalize_image(img)

    # Apply mask if provided
    if mask is not None:
        img = np.ma.array(img, mask=(mask==0))

    # Determine clipping values
    vmin, vmax = image_clip_values(img, vmin, vmax, minPct, maxPct)

    # Spawn figure and axis if not given
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Plot image
    cax = ax.imshow(img, extent=extent,
        cmap=cmap, vmin=vmin, vmax=vmax)

    # Colorbar
    if cbarOrient == 'auto':
        # Orient colorbar based on image dimensions
        M, N = img.shape
        if M > N:
            cbarOrient = 'vertical'
        elif N >= M:
            cbarOrient = 'horizontal'

    if cbarOrient is not None and equalize is False:
        fig.colorbar(cax, ax=ax, orientation=cbarOrient)

    return fig, ax
# <<<<<<


def raster_multiplot(imgs, mrows=1, ncols=1,
        mask=None, extent=None,
        cmap='viridis', cbarOrient=None,
        vmin=None, vmax=None, minPct=None, maxPct=None,
        titles=None, suptitle=None,
        fig=None, axes=None):
    '''
    Plot multiple raster data sets.
    Inherits plot_raster.
    '''
    # Setup
    MN = mrows * ncols  # figure dimensions

    # Checks
    if titles is not None and len(titles) > 1:
        assert len(titles) == len(imgs), \
            'Number of titles must equal the number of images or constitute a single supertitle'

    # Spawn figure and axis if not given
    if fig is None and axes is None:
        fig, axes = plt.subplots(nrows=mrows, ncols=ncols)
    
    if mrows > 1: axes = [ax for row in axes for ax in row]

    # Loop through images to plot
    k = 0
    for i, img in enumerate(imgs):
        # Spawn
        if (MN - k) == 0:
            # Update old figure
            fig.suptitle(suptitle)
            fig.tight_layout()

            # Spawn new figure
            fig, axes = plt.subplots(nrows=mrows, ncols=ncols)
            if mrows > 1: axes = [ax for row in axes for ax in row]
            fig.suptitle(suptitle)
            k = 0  # reset counter

        # Plot image
        fig, axes[k] = plot_raster(img, mask=mask, extent=extent,
            cmap=cmap, cbarOrient=cbarOrient,
            vmin=vmin, vmax=vmax, minPct=minPct, maxPct=maxPct,
            fig=fig, ax=axes[k])

        # Format plot
        if titles is not None: axes[k].set_title(titles[i])

        # Update counter
        k += 1

    # Format final figure
    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes



### HEAT MAPS ---
def histogram2d(xData, yData, cmap='viridis', cbarOrient='horizontal', nbins=30, logDensity=False, fig=None, ax=None):
    ''' Plot a 2D histogram of the 1D data sets xData and yData. '''
    # Construct histogram
    H, xedges, yedges = np.histogram2d(xData, yData, bins=nbins)

    # Format arrays
    H = H.T  # transpose map
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    # Spawn figure and axis if not given
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Color normalization
    if logDensity == True:
        colorNorm = pltColors.LogNorm()
    else:
        colorNorm = None

    # Plot histogram
    cax = ax.pcolormesh(X, Y, H, cmap=cmap, norm=colorNorm, shading='gouraud')

    # Format colorbar
    fig.colorbar(cax, ax=ax, orientation=cbarOrient)


def kde2d(xData, yData, plotType='pcolormesh', cmap='viridis', cbarOrient='horizontal',
        nbins=30, logDensity=False, fig=None, ax=None):
    '''
    Plot a 2D kernel density estimate of the 1D data sets xData and yData.
    Different plot types are available, including 'pcolormesh', 'contour',
     and 'contourf'.
    '''
    # Construct grid
    x = np.linspace(xData.min(), xData.max(), nbins)
    y = np.linspace(yData.min(), yData.max(), nbins)
    X, Y = np.meshgrid(x, y)

    # Construct KDE
    positions = np.vstack([X.flatten(), Y.flatten()])
    values = np.vstack([xData, yData])
    kernel = gaussian_kde(values)
    H = kernel(positions)
    H = H.reshape(X.shape)

    # Spawn figure and axis if not given
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Color normalization
    if logDensity == True:
        colorNorm = pltColors.LogNorm()
    else:
        colorNorm = None

    # Plot KDE
    if plotType == 'pcolormesh':
        # Plot KDE only
        cax = ax.pcolormesh(X, Y, H, cmap=cmap, norm=colorNorm)

    elif plotType == 'contour':
        # Plot unfilled contours
        cax = ax.contour(X, Y, H, cmap=cmap, norm=colorNorm)

    elif plotType == 'contourf':
        # Plot filled contours
        cax = ax.contourf(X, Y, H, cmap=cmap, norm=colorNorm)

    # Format colorbar
    fig.colorbar(cax, ax=ax, orientation=cbarOrient)



### VECTORS ---
def plot_look_vectors(Px, Py, Pz):
    ''' Plot look vectors based on ARIA or ISCE convention. '''
    # Horizontal component
    Ph = np.linalg.norm([Px, Py])

    # Spawn figure
    fig, [axInc, axAz] = plt.subplots(ncols=2)

    # Plot incidence
    axInc.quiver(0, 0, Ph*np.sign(Px), Pz, color='k', units='xy', scale=1, zorder=2)

    # Plot incidence reference lines
    axInc.axhline(0, color=(0.2, 0.8, 0.5), zorder=1)
    axInc.axvline(0, color=[0.7]*3, linestyle='--', zorder=1)

    # Format incidence axis
    axInc.set_xlim([-1, 1])
    axInc.set_ylim([-0.1, 1])
    axInc.set_aspect(1)
    axInc.set_title('Incidence')

    # Plot azimuth
    axAz.quiver(0, 0, Px, Py, color='k', units='xy', scale=1, zorder=2)

    # Plot azimuth referene lines
    axAz.axhline(0, color=[0.7]*3, zorder=1)
    axAz.axvline(0, color=[0.7]*3, zorder=1)

    # Format azimuth axis
    axAz.set_xlim([-1, 1])
    axAz.set_ylim([-1, 1])
    axAz.set_aspect(1)
    axAz.set_title('Azimuth')

    fig.tight_layout()

    return fig, [axInc, axAz]



### PROFILES ---
def plot_map_profile(profGeom, fig=None, ax=None):
    '''
    Plot a map profile based on the corners provided by the "profile_geometry"
     class.
    '''
    # Spawn figure if necessary
    if fig is None and ax is None:
        fig, ax = plt.subplots()

    # Plot profile
    ax.fill(profGeom.corners[:,0], profGeom.corners[:,1],
        facecolor=(0.5, 0.5, 0.5), edgecolor='k', alpha=0.5)

    # Format axis
    ax.set_aspect(1)

    return fig, ax
