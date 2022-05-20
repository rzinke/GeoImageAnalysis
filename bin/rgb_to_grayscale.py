#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Create a new raster by doing math on existing rasters.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ImageIO import load_gdal_dataset, confirm_outdir, confirm_outname_ext, save_gdal_dataset
from ImageMasking import mask_dataset
from GeoFormatting import DS_to_extent
from ImageViewing import plot_raster


### PARSER ---
Description = '''Convert an RGB(A) image to gray.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    # Inputs
    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='rgbImg', type=str,
        help='Name of RGB(A) image.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')

    # Outputs
    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot outputs.')
    OutputArgs.add_argument('--plot-inputs', dest='plotInputs', action='store_true',
        help='Plot inputs.')
    OutputArgs.add_argument('-o','--outName', dest='outName', default='Gray',
        help='Name of output file.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### LOADING ---
def load_dataset(rgbImg, verbose=False):
    ''' Load image and confirm formatting. '''
    # Load GDAL data set
    DS = load_gdal_dataset(rgbImg, verbose=verbose)

    # Confirm three or four bands
    nBands = DS.RasterCount
    assert nBands >= 3, \
        'Only {:d} bands detected. Provide 3+ band image with RGB.'.format(nBands)

    return DS


def retreive_bands(DS, verbose=False):
    ''' Retreive RGB bands. '''
    # Retreive bands
    rImg = DS.GetRasterBand(1).ReadAsArray()
    bImg = DS.GetRasterBand(2).ReadAsArray()
    gImg = DS.GetRasterBand(3).ReadAsArray()

    # Alpha
    if DS.RasterCount > 3:
        aImg = DS.GetRasterBand(4).ReadAsArray()
    else:
        aImg = np.ones((DS.RasterYSize, DS.RasterXSize))

    return rImg, bImg, gImg, aImg



### PLOTTING ---
def plot_inputs(rImg, bImg, gImg, mask, extent=None):
    ''' Plot input data sets and mask. '''
    # Spawn figure
    inpsFig, axInps = plt.subplots(ncols = 4)

    # Plot bands
    plot_raster(rImg, mask=mask, extent=extent,
        cmap='Reds', cbarOrient='auto',
        fig=inpsFig, ax=axInps[0])
    axInps[0].set_title('Red')

    plot_raster(gImg, mask=mask, extent=extent,
        cmap='Greens', cbarOrient='auto',
        fig=inpsFig, ax=axInps[1])
    axInps[1].set_title('Blue')
    axInps[1].set_yticks([])

    plot_raster(bImg, mask=mask, extent=extent,
        cmap='Blues', cbarOrient='auto',
        fig=inpsFig, ax=axInps[2])
    axInps[2].set_title('Green')
    axInps[2].set_yticks([])

    plot_raster(aImg, mask=mask, extent=extent,
        cmap='cividis', cbarOrient='auto',
        fig=inpsFig, ax=axInps[3])
    axInps[3].set_title('Alpha')
    axInps[3].set_yticks([])


def plot_output(grayImg, mask, extent):
    ''' Plot the output dataset. '''
    # Plot raster
    outFig, axOut = plot_raster(grayImg, mask=mask, extent=extent,
        cmap='Greys_r', cbarOrient='auto')

    # Format plot
    axOut.set_title('Grayscale')



### RGB TO GRAYSCALE ---
def convert_rgb_to_gray(rImg, gImg, bImg, verbose=False):
    '''
    Convert red, blue, and green bands to grayscale using the formula
        "R*0.2989+G*0.5870+B*0.1140"
    found here:
    https://gis.stackexchange.com/questions/270144/how-to-convert-4band-rgb-a-geotiff-to-1band-gray-geotiif
    '''
    if verbose == True: print('Converting to grayscale')

    # Color mixing factors
    Rfactor = 0.2989
    Gfactor = 0.5870
    Bfactor = 0.1140

    if verbose == True: print('\tR * {:.4f}\n\tG * {:.4f}\n\tB * {:.4f}'.format(Rfactor, Gfactor, Bfactor))

    # Mix colors
    grayImg = Rfactor*rImg + Gfactor*gImg + Bfactor * bImg

    return grayImg



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load and format data
    # Load data sets
    DS = load_dataset(inps.rgbImg, verbose=inps.verbose)

    # Geographic extent
    extent = DS_to_extent(DS, verbose=inps.verbose)

    # Determine mask
    # Create master mask for data set
    mask = mask_dataset(DS, inps.maskArgs, verbose=inps.verbose)

    # Retreive bands
    rImg, bImg, gImg, aImg = retreive_bands(DS, verbose=inps.verbose)

    # Plot inputs if requested
    if inps.plotInputs == True: plot_inputs(rImg, bImg, gImg, mask, extent)


    ## Convert to grayscale
    if inps.verbose == True: print('*'*32)  # stdout break

    # Color conversion
    grayImg = convert_rgb_to_gray(rImg, bImg, gImg,verbose=inps.verbose)


    ## Save to file
    # Checks
    confirm_outdir(inps.outName)  # confirm output directory exists
    outName = confirm_outname_ext(inps.outName, ['tif', 'tiff'])  # confirm file extension

    # Save data set
    save_gdal_dataset(outName, grayImg, mask=mask, exDS=DS, verbose=inps.verbose)


    ## Plot
    if inps.plot == True: plot_output(grayImg, mask, extent)


    plt.show()