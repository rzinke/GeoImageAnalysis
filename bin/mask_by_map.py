#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Mask the values in a data set based on the values in a similar data set.
E.g., mask velocity values by temporal or spatial coherence.

FUTURE IMPROVEMENTS

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import argparse
import os
import matplotlib.pyplot as plt

from ImageIO import load_gdal_dataset, confirm_outdir, confirm_outname_ext, save_gdal_dataset
from RasterResampling import match_rasters
from GeoFormatting import DS_to_extent
from ImageViewing import plot_raster


### PARSER ---
Description = '''Mask the values in a data set based on the values in a similar data set.
E.g., mask velocity values based on spatial or temporal coherence.'''

Examples = '''Mask by temporal coherence values less than 0.7
mask_by_map.py velocity.tif --mask-layer temporalCoherence.tif --mask-value 0.7 --mask-operator < -o velocity_masked.tif'''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')

    InputArgs.add_argument(dest='imgName', type=str,
        help='Name of data set to mask.')
    InputArgs.add_argument('--mask-layer', dest='maskLayer', type=str, required=True,
        help='Mask layer.')
    InputArgs.add_argument('--mask-value', dest='maskValue', type=float, required=True,
        help='Mask value.')
    InputArgs.add_argument('--mask-operator', dest='maskOperator', type=str, default='==',
        help='Mask operator. ([=], >, <). E.g., < 0.7 masks values less than 0.7.')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-o','--outName', dest='outName', type=str, default='Out', 
        help='Output name.')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true', 
        help='Plot inputs and outputs.')
    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


### LOADING ---
def load_datasets(imgName, maskLayer, verbose=False):
    '''
    Load the GDAL data sets containing the image to be masked and the mask layer.
    Resample the data sets to the same area and resolution.
    '''
    if verbose == True: print('Loading data sets')

    # Load image to be masked
    if verbose == True: print('Image to be masked: {:s}'.format(imgName))
    DS = load_gdal_dataset(imgName, verbose=verbose)

    # Load mask layer
    if verbose == True: print('Mask layer: {:s}'.format(maskLayer))
    MDS = load_gdal_dataset(maskLayer, verbose=verbose)


    ## Crop to same extent
    # Group data sets
    datasets = {'img':DS, 'msk':MDS}

    # Resample to same extent and resolution
    datasets = match_rasters(datasets,
                cropping='intersection', resolution='fine',
                verbose=verbose)

    return datasets



### MASKING ---
def mask_map(img, msk, maskValue, maskOperator, verbose=False):
    '''
    Apply the mask based on the mask layer and the specified value and operator.
    '''
    if verbose == True: print('Applying mask')

    # Setup
    maskedImg = img.copy()

    # Apply based on condition
    if inps.maskOperator in ['=', '==']:
        if verbose == True:
            print('Apply condition img[msk == {:f}] = 0'.format(maskValue))
        maskedImg[msk == maskValue] = 0
    elif inps.maskOperator in ['<']:
        if verbose == True:
            print('Apply condition img[msk < {:f}] = 0'.format(maskValue))
        maskedImg[msk < maskValue] = 0
    elif inps.maskOperator in ['>']:
        if verbose == True:
            print('Apply condition img[msk > {:f}] = 0'.format(maskValue))
        maskedImg[msk > maskValue] = 0
    else:
        print('Condition {:s} not recognized. Use =, <, >'.format(maskOperator))

    return maskedImg



### SAVING ---
## Saving
def save(outName, maskedImg, exDS, verbose=False):
    '''
    Save the vector components fields to a three-band GeoTiff using GDAL.
    '''
    if verbose == True: print('Saving vector fields')

    # Check outname
    confirm_outdir(outName)  # confirm output directory exists
    outName = confirm_outname_ext(outName, ['.tif'])  # confirm output extension if GeoTiff

    # Save to GDAL data set
    save_gdal_dataset(outName, maskedImg, exDS=exDS, verbose=verbose)



### PLOTTING ---
def plot(img, msk, maskedImg):
    '''
    Plot the original image, the masking layer, and the masked image.
    '''
    # Spawn figure and axes
    fig, [axImg, axMsk, axMasked] = plt.subplots(figsize=(9, 6), ncols=3)
    cbarOrient = 'horizontal'

    # Plot original image
    plot_raster(img, extent=extent,
        cmap='jet', cbarOrient=cbarOrient, minPct=1, maxPct=99,
        fig=fig, ax=axImg)

    # Plot mask layer
    plot_raster(msk, extent=extent,
        cmap='viridis', cbarOrient=cbarOrient, minPct=1, maxPct=99,
        fig=fig, ax=axMsk)

    # Plot masked image
    plot_raster(maskedImg, extent=extent,
        cmap='jet', cbarOrient=cbarOrient, minPct=1, maxPct=99,
        fig=fig, ax=axMasked)




### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load images
    datasets = load_datasets(inps.imgName, inps.maskLayer, verbose=inps.verbose)


    ## Extract images and metadata
    extent = DS_to_extent(datasets['img'])
    img = datasets['img'].GetRasterBand(1).ReadAsArray()
    msk = datasets['msk'].GetRasterBand(1).ReadAsArray()


    ## Apply mask
    maskedImg = mask_map(img, msk, inps.maskValue, inps.maskOperator,
        verbose=inps.verbose)


    ## Save
    save(inps.outName, maskedImg, exDS=datasets['img'], verbose=inps.verbose)


    ## Plot
    if inps.plot == True:
        plot(img, msk, maskedImg)


        plt.show()
