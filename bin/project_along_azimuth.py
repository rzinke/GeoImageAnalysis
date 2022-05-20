#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Project a two-component image along a specified azimuth.

FUTURE IMPROVEMENTS


TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ImageIO import confirm_outdir, confirm_outname_ext, load_gdal_datasets, save_gdal_dataset
from GeoFormatting import DS_to_extent
from RasterResampling import match_rasters
from ImageMasking import create_common_mask
from ImageViewing import plot_raster


### PARSER ---
Description = '''Project the EW and NS components of an image data set along the specified azimuth.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument('-e','--east-img', dest='EimgFile', type=str, required=True,
        help='East image file name.')
    InputArgs.add_argument('-n','--north-img', dest='NimgFile', type=str, required=True,
        help='North image file name.')
    InputArgs.add_argument('-a','--azimuth', dest='projAzimuth', type=float, required=True,
        help='Projection azimuth, deg clockwise from north.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')


    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outName', type=str, default='Proj',
        help='GPS-corrected field.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### LOADING ---
def load_datasets(dsFiles, verbose=False):
    '''
    Load the GDAL-compatible data sets and crop to same bounds/resolution.
    '''
    if verbose == True: print('Loading and formatting input data sets')

    # Load data sets using GDAL
    datasets = load_gdal_datasets(dsFiles, dsNames=['EW', 'NS'], verbose=inps.verbose)

    # Check that spatial extents are same
    Eextent = DS_to_extent(datasets['EW'], verbose=verbose)
    Nextent = DS_to_extent(datasets['NS'], verbose=verbose)

    if Eextent != Nextent:
        print('Extents not equal ... resampling.')
        datasets = match_rasters(datasets, cropping='intersection', resolution='coarse', verbose=verbose)

    return datasets


def format_images(datasets, mask, verbose=False):
    '''
    Format the image before filtering to replace NaNs and masked values with
     zeros.
    '''
    if verbose == True: print('Pre-formatting images')

    # Extract images
    EWimg = datasets['EW'].GetRasterBand(1).ReadAsArray()
    NSimg = datasets['NS'].GetRasterBand(1).ReadAsArray()

    # Replace NaNs with zeros
    EWimg[np.isnan(EWimg) == 1] = 0
    NSimg[np.isnan(NSimg) == 1] = 0

    # Replace mask values
    EWimg[mask==0] = 0
    NSimg[mask==0] = 0

    return EWimg, NSimg



### PROJECTION ---
def project_imgs(EWimg, NSimg, az, verbose=False):
    ''' Project images along azimuth. '''
    if verbose == True: print('Projecting images along azimuth {:03.1f}'.format(az))    

    # Parameters
    assert EWimg.shape == NSimg.shape, \
        'EWimg and NSimg are different sizes: ({:d} x {:d}) vs ({:d} x {:d})'.format(EWimg.shape, NSimg.shape)
    M, N = EWimg.shape

    # Convert azimuth angle into unit vector
    az = 90-az  # Cartesian coordinates
    az = np.deg2rad(az)  # convert to radians
    u = np.array([[np.cos(az), np.sin(az)]])  # projection vector
    u = u/np.linalg.norm(u)  # unit length

    # Stack data values
    D = np.vstack([EWimg.flatten(), NSimg.flatten()])

    # Project along azimuth
    P = np.dot(u, D)
    P = P.reshape(M, N)

    # Report if requested
    if verbose == True: print('Unit vector: [{:f} {:f}]'.format(*u.flatten()))

    return P



### OUTPUTS ---
def save_projected(outName, img, mask, exDS, verbose=False):
    ''' Save the projected image to a GeoTIFF format. '''
    # Format output name
    outName = confirm_outname_ext(outName, ext=['tif'], verbose=verbose)
    confirm_outdir(outName)

    # Save to file
    save_gdal_dataset(outName, img, mask, exDS=exDS, verbose=verbose)


def plot_results(EWimg, NSimg, Pimg, mask, extent, verbose=False):
    ''' Plot the input and output images. '''
    if verbose == True: print('Plotting inputs and outputs')

    # Parameters
    cbarOrient = 'auto'

    # Plot for input images
    fig0, [axEW, axNS] = plt.subplots(ncols=2)

    # Plot EW image
    fig0, axEW = plot_raster(EWimg, mask=mask, extent=extent,
        cmap='bwr', cbarOrient=cbarOrient,
        minPct=1, maxPct=99,
        fig=fig0, ax=axEW)
    axEW.set_title('Original EW image')

    # Plot NS image
    fig0, axNS = plot_raster(NSimg, mask=mask, extent=extent,
        cmap='bwr', cbarOrient=cbarOrient,
        minPct=1, maxPct=99,
        fig=fig0, ax=axNS)
    axNS.set_title('Original NS image')


    # Plot projected image
    figP, axP = plot_raster(Pimg, mask=mask, extent=extent,
        cmap='bwr', cbarOrient=cbarOrient,
        minPct=1, maxPct=99)
    axP.set_title('Projected image')   



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Load data
    # Load image data sets
    datasets = load_datasets([inps.EimgFile, inps.NimgFile], verbose=inps.verbose)

    # Create masks
    mask = create_common_mask(datasets, inps.maskArgs, verbose=inps.verbose)

    # Retrieve extent
    extent = DS_to_extent(datasets['EW'], verbose=inps.verbose)

    # Pre-format images
    EWimg, NSimg = format_images(datasets, mask, verbose=inps.verbose)

    ## Projection
    Pimg = project_imgs(EWimg, NSimg, inps.projAzimuth, verbose=inps.verbose)


    ## Outputs
    # Save to file
    save_projected(outName=inps.outName, img=Pimg, mask=mask, exDS=datasets['EW'],
        verbose=inps.verbose)

    # Plot if requested
    plot_results(EWimg, NSimg, Pimg, mask, extent, verbose=inps.verbose)


    plt.show()