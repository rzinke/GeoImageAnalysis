#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Apply a principal component analysis to an image data set.

FUTURE IMPROVEMENTS

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from ImageIO import load_gdal_datasets
from ImagePCA import StackPCA



### PARSER ---
Description = '''Apply a principal component analysis to a series of images.'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='datasets', nargs='+', type=str,
        help='Name of reference image.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')

    AnalysisArgs = parser.add_argument_group('ANALYSIS')
    AnalysisArgs.add_argument('--no-standardize', dest='noStandardize', action='store_false',
        help='Do NOT subtract mean from data sets.')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('--cmap', dest='cmap', type=str, default='jet',
        help='Colormap for plots. ([jet]).')
    OutputArgs.add_argument('--plot-inputs', dest='plotInputs', action='store_true',
        help='Plot input data.')
    OutputArgs.add_argument('--plot-retention', dest='plotRetention', action='store_true',
        help='Plot variance retention.')
    OutputArgs.add_argument('-p','--plotPCs', dest='plotPCs', type=int, default=None,
        help='Plot components. (Provide number of components to plot).')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default=None,
        help='Principal components name.')
    OutputArgs.add_argument('--pcs-to-save', dest='PCsToSave', type=int, default=None,
        help='Number of principal components to save as maps. ([None = all]).')

    return parser


def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### PCA ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Format image list
    # Find images
    if len(inps.datasets) == 1:
        # Format search string
        searchStr = inps.datasets[0]

        # Find list of data sets
        datasets = glob(searchStr)
    else:
        datasets = inps.datasets

    # Sort into alphabetical order
    datasets.sort()


    ## PCA
    # Instantiate object
    pca = StackPCA(verbose=inps.verbose)

    # Load data
    pca.load_images(datasets, maskArgs=inps.maskArgs)

    # Plot input images
    if inps.plotInputs == True: pca.plot_inputs(cmap=inps.cmap)

    # Apply PCA
    pca.run_pca(standardize=inps.noStandardize)

    # Plot variation retention
    if inps.plotRetention == True: pca.plot_retention()

    # Plot PCs
    if inps.plotPCs is not None:
        pca.plot_components(inps.plotPCs, cmap=inps.cmap)

    # Save maps
    if inps.outname is not None:
        if inps.PCsToSave is None: inps.PCsToSave = 'all'
        pca.save(inps.outname, inps.PCsToSave)


    plt.show()
