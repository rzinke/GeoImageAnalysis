#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Compute a 1D timeseries for the specified data set.

FUTURE IMPROVEMENTS


TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

from Timeseries import Timeseries
from ImageMasking import create_common_mask
from ImageViewing import plot_raster


### PARSER ---
Description = '''Invert for the displacement timeseries for a sequence of images.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument('-d','--date-list', dest='datePairList', nargs='+', type=str, required=True,
        help='List of date pairs (<secondary>-<master>) corresponding to the difference maps.')
    InputArgs.add_argument('-i','--images', dest='imgList', nargs='+', type=str, required=True,
        help='List of images or folder/wildcard.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')
    InputArgs.add_argument('-b','--band', dest='bandNb', type=int, default=1,
        help='Image band number to display.')


    InversionArgs = parser.add_argument_group('INVERSION')
    InversionArgs
    InversionArgs.add_argument('--regularize', dest='regularize', action='store_true',
        help='Apply regularization equations assuming constant velocity.')
    InversionArgs.add_argument('--regularization-weight', dest='regularizationWeight', type=float, default=1.0,
        help='Regularization equations weight. (0, 0.1, 0.2, ..., [1.0], ...')


    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str,
        help='GPS-corrected field.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
def load_date_pairs(datePairList, verbose=False):
    '''
    Load the date pairs from a single file, or as a list passed as an argument.
    If the date pair list has only one item, assume that is a text file with a
     list of date pairs.
    If input list is multiple items, assume those items are date pairs.
    '''
    if verbose == True: print('Loading date pairs')

    # Number of items in list
    n = len(datePairList)

    if n == 1:
        # Assume file list of date pairs
        if verbose == True:
            print('... 1 item detected. Assuming list of date pairs.')

        # Read date pairs from file
        with open(datePairList[0], 'r') as pairFile:
            lines = pairFile.readlines()
            datePairs = [line.strip('\n') for line in lines]
    else:
        # Assume date pairs passed as arguments
        if verbose == True:
            print('... {:d} items detected. Assuming date pairs passed via command line.'.\
                format(n))

        # Parse date pairs
        datePairs = datePairList

    # Report if requested
    print('... {:d} date pairs identified'.format(len(datePairs)))

    return datePairs


def parse_img_list(imgList, verbose=False):
    '''
    Parse the supplied image list into a list of images.
    If the image list is only one item, assume the input is a folder with image
     files in tiff format or a list of file names in a text document.
    If the image list is multiple items, assume those items are images.
    '''
    if verbose == True: print('Loading images')

    # Number of items in list
    n = len(imgList)

    if n == 1 and not imgList[0].endswith('.txt'):
        # Assume input is a directory with files
        if verbose == True:
            print('... 1 item detected. Assuming folder of images.')

        # Read date pairs from file
        imgList = glob(imgList[0])

    elif n == 1 and imgList[0].endswith('.txt'):
        # Assume input is a list of file names in a text document
        if verbose == True:
            print('... text file provided. Assuming list of file names.')

        # Load file names from text file
        with open(imgList[0], 'r') as imgFile:
            # Read lines from file
            imgList = imgFile.readlines()

            # Format lines
            imgList = [imgName.strip('\n') for imgName in imgList]

    elif n > 1:
        # Assume images passed as arguments
        if verbose == True:
            print('... {:d} items detected. Assuming images passed via command line.'.\
                format(n))

    # Report if requested
    print('... {:d} images identified'.format(len(imgList)))

    return imgList



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Format inputs
    # Load date pairs
    datePairs = load_date_pairs(inps.datePairList, verbose=inps.verbose)

    # Parse list of images
    imgList = parse_img_list(inps.imgList, verbose=inps.verbose)


    ## Timeseries
    # Instantiate object
    ts = Timeseries(verbose=inps.verbose)

    # Load data
    ts.set_date_pairs(datePairs)
    ts.load_measurements(imgList, maskArgs=inps.maskArgs, band=inps.bandNb)

    # Compute timeseries
    ts.compute_timeseries(regularize=inps.regularize,
        regularizationWeight=inps.regularizationWeight)

    # Plot if requested
    if inps.plot == True:
        ts.plot()

    # Save to file
    if inps.outname is not None:
        ts.save(inps.outname)


plt.show()
