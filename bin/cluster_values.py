#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Apply a k-means cluster algorithm to the given image.

FUTURE IMPROVEMENTS
    Masking


TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from ImageIO import confirm_outdir, confirm_outname_ext, load_gdal_dataset, save_gdal_dataset
from GeoFormatting import DS_to_extent
from ImageClassification import KmeansClustering
from ImageViewing import image_stats, image_percentiles, plot_raster


### PARSER ---
Description = '''Apply a k-means cluster algorithm to the given image.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='imgFile', type=str,
        help='Image file name.')
    InputArgs.add_argument('-b','--band', dest='bandNb', type=int, default=1,
        help='Image band number to display.')

    ClusterArgs = parser.add_argument_group('CLUSTERING')
    ClusterArgs.add_argument('-k','--k-clusters', dest='kClusters', type=int, default=2,
        help='Number of clusters. (1, [2], ...).')
    ClusterArgs.add_argument('-i','--max-iterations', dest='maxIterations', type=int, default=20,
        help='Maximum number of iterations. (1, 2, ..., [20], ...)')
    ClusterArgs.add_argument('-c','--init-centroids', dest='initCentroids', nargs='+', default='auto',
        help='Initial centroid locations. ([\'auto\'])')

    PlottingArgs = parser.add_argument_group('PLOTTING')
    PlottingArgs.add_argument('--cmap', dest='cmap', type=str, default='viridis',
        help='Plot colormap. ([viridis])')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default='Clustered',
        help='Clustered values map name.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### AUXILIARY FUNCTIONS ---
def init_centroids(centroids0, kClusters, verbose=False):
    '''
    Initial centroids argument must be auto, or same number of arguments
     as clusters.
    '''
    # Number of centroids
    nCentroids = len(centroids0)

    # Check/format centroid input(s)
    if not centroids0 == 'auto':
        # Check number of inputs
        assert nCentroids == kClusters, \
            'Number of centroid inputs ({:d}) must equal the number of clusters ({:d})'.\
            format(nCentroids, kClusters)

        # Format inputs
        centroids0 = [float(centroid) for centroid in centroids0]
        centroids0.sort()

        # Report if requested
        if verbose == True:
            print('{:d} initial centroids provided ({:f} - {:f})'.\
                format(nCentroids, centroids0[0], centroids0[-1]))

    else:
        if verbose == True:
            print('Initial centroids will be automatically generated')

    return centroids0


def plot_images(origImg, clusteredImg, extent=None, cmap='viridis'):
    ''' Plot the original and clustered images. '''
    # Spawn figure and axes
    fig, axes = plt.subplots(figsize=(8,4), ncols=2)

    # Plot original image
    plot_raster(origImg, mask=None, extent=extent,
    cmap=cmap, cbarOrient='auto',
    fig=fig, ax=axes[0])

    axes[0].set_title('Original image')

    # Plot clustered image
    plot_raster(clusteredImg, mask=None, extent=extent,
    cmap=cmap, cbarOrient='auto',
    fig=fig, ax=axes[1])

    axes[1].set_title('Clustered image')

def plot_centroids(origImg, centroids):
    '''
    Plot the cluster centroids against the histogram of original image values.
    '''
    # Downsample image
    maxValues = 1E4
    skips = int(np.multiply(*origImg.shape)/maxValues)
    dsampImg = origImg.flatten()[::skips]

    # Range of values
    valueRng = np.linspace(dsampImg.min(), dsampImg.max(), 1000)

    # Create histogram of image values
    kde = gaussian_kde(dsampImg)
    hist = kde(valueRng)

    # Spawn figure and axis
    fig, ax = plt.subplots(figsize=(7,3))

    # Plot histogram and centroids
    ax.plot(valueRng, hist, color='b', linewidth=2, label='orig values')
    for k, centroid in enumerate(centroids):
        ax.axvline(centroid, color='r', linewidth=2,
            label='centroid {:d}'.format(k+1))

    # Format figure
    ax.set_title('Histogram and centroids')
    ax.legend()



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load and format inputs
    # Check centroid initialization
    centroids0 = init_centroids(inps.initCentroids, inps.kClusters,
        verbose=inps.verbose)

    # Load image data set
    DS = load_gdal_dataset(inps.imgFile, verbose=inps.verbose)
    extent = DS_to_extent(DS)
    M, N = DS.RasterYSize, DS.RasterXSize

    # Retreive and format image
    origImg = DS.GetRasterBand(inps.bandNb).ReadAsArray()
    origImg = origImg.reshape(M*N, 1)


    ## K-means cluster analysis
    # Cluster analysis
    clusters = KmeansClustering(origImg,
            inps.kClusters, centroids=centroids0,
            maxIterations=inps.maxIterations,
            verbose=inps.verbose)

    # Extract clusterd image values
    clusteredImg = origImg.copy()
    for k in range(inps.kClusters):
        clusteredImg[clusters.centroidIndices==k] = clusters.centroids[k]

    # Reformat images
    origImg = origImg.reshape(M,N)
    clusteredImg = clusteredImg.reshape(M,N)


    ## Outputs
    # Plot if requested
    if inps.plot == True:
        if inps.verbose == True:
            print('Plotting')

        # Plot original and clustered images
        plot_images(origImg, clusteredImg, extent=extent, cmap=inps.cmap)

        # Plot histogram and cluster centroids
        plot_centroids(origImg, clusters.centroids)

    # Save to file
    outname = confirm_outname_ext(inps.outname, ext=['tif'])
    confirm_outdir(inps.outname)
    save_gdal_dataset(outname, clusteredImg, exDS=DS,
        verbose=inps.verbose)


    plt.show()
