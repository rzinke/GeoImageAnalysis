#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Apply a k-means cluster algorithm to the given image.

FUTURE IMPROVEMENTS
    Masking needs to be improved for multi-band imagery


TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from ImageIO import confirm_outdir, confirm_outname_ext, load_gdal_dataset, images_from_dataset, save_gdal_dataset
from ImageMasking import create_mask
from GeoFormatting import DS_to_extent
from ImageClassification import KmeansClustering
from ImageViewing import plot_histogram, plot_raster


### PARSER ---
Description = '''Apply a k-means cluster algorithm to the given image.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='imgFile', type=str,
        help='Image file name.')
    InputArgs.add_argument('-b','--bands', dest='bands', nargs='+', default='all',
        help='Image band number to display.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')

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
    OutputArgs.add_argument('--plot-inputs', dest='plotInputs', action='store_true',
        help='Plot input bands.')
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


def format_images(DS, bands='all', maskArgs=None, verbose=False,
    plotBands=False):
    '''
    Extract the image bands from the provided data set.
    Plot the images if requested.
    Format the images for cluster analysis by flattening the data into
     an (mDatapoints x nDimensions) matrix.
    '''
    # Extract bands as M x N arrays
    bands = images_from_dataset(DS, bands=bands, verbose=verbose)
    nBands = len(bands)

    if verbose == True:
        print('Extracted {:d} out of {:d} bands'.\
            format(nBands, DS.RasterCount))

    # Determine mask
    mask = np.ones((DS.RasterYSize, DS.RasterXSize))
    for band in bands:
        mask *= create_mask(band, maskArgs)

    # Plot input bands if requested
    if plotBands == True:
        for i in range(nBands):
            # Spawn figure and axis
            inptFig, inptAx = plt.subplots()

            # Plot band
            plot_raster(bands[i], mask=mask,
                cmap='Greys', cbarOrient='auto',
                fig=inptFig, ax=inptAx)

            # Format plot
            inptAx.set_title('Band {:d}'.format(i+1))
            inptAx.set_xticks([])
            inptAx.set_yticks([])

    # Flatten arrays to (MN x nBands)
    imgValues = np.column_stack([band.flatten() for band in bands])
    mask = mask.flatten()
    del bands

    # Report if requested
    if verbose == True:
        print('Images formatted into {:d} x {:d} matrix'.\
            format(*imgValues.shape))

    return imgValues, mask


def plot_centroids(imgValues, mask, centroids):
    '''
    Plot the cluster centroids against the histogram of original image values.
    '''
    # Determine number of images
    nBands = imgValues.shape[1]
    kCentroids = len(centroids)

    # Spawn figure and axes
    centroidFig, centroidAxes = plt.subplots(nrows=nBands)
    if nBands == 1:
        # Needs to be list if not one already
        centroidAxes = [centroidAxes]

    # Plot each
    for i in range(nBands):
        # Plot histogram of image values
        plot_histogram(imgValues[mask==1,i],
            plotType='kde', nBins=1000,
            fig=centroidFig, ax=centroidAxes[i])

        # Plot centroids
        for k in range(kCentroids):
            centroidAxes[i].axvline(centroids[k,i],
                color='r', linewidth=2)


def plot_images(origImg, clusteredImg, mask):
    ''' Plot the original and clustered images. '''
    # Spawn figure and axes
    imgFig, imgAxes = plt.subplots(figsize=(8,4), ncols=2)

    # Plot
    imgAxes[0].imshow(origImg)
    imgAxes[0].set_title('Original image')
    imgAxes[0].set_xticks([])
    imgAxes[0].set_yticks([])


    imgAxes[1].imshow(clusteredImg.astype(int))
    imgAxes[1].set_title('Clustered image')
    imgAxes[1].set_xticks([])
    imgAxes[1].set_yticks([])



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

    # Format images
    imgValues, mask = format_images(DS,
            bands=inps.bands,
            maskArgs=inps.maskArgs,
            verbose=inps.verbose,
            plotBands=inps.plotInputs)
    nBands = imgValues.shape[1]


    ## K-means cluster analysis
    # Cluster analysis
    clusters = KmeansClustering(imgValues[mask==1,:],
            inps.kClusters, centroids=centroids0,
            maxIterations=inps.maxIterations,
            verbose=inps.verbose)

    # Classify image values
    clusteredImg = clusters.classify(imgValues)
    clusteredImg = clusteredImg.reshape(M, N, nBands)


    ## Outputs
    # Plot if requested
    if inps.plot == True:
        if inps.verbose == True:
            print('Plotting')

        # Plot histogram and cluster centroids
        plot_centroids(imgValues, mask, clusters.centroids)

        # Plot original and clustered images
        plot_images(imgValues.reshape(M, N, nBands),
            clusteredImg,
            mask=mask.reshape(M, N))

    # Save to file
    outname = confirm_outname_ext(inps.outname, ext=['tif'])
    confirm_outdir(inps.outname)
    save_gdal_dataset(outname, [clusteredImg[:,:,i] for i in range(nBands)],
        exDS=DS, verbose=inps.verbose)


    plt.show()
