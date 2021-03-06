'''
SHORT DESCRIPTION
Determine a single, binary mask given an image and one or more mask values.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import os
import numpy as np
from osgeo import gdal
from scipy.stats import mode

from DatasetChecks import check_dataset_sizes
from RasterResampling import gdal_resample



### DETECT BACKGROUND ---
def detect_background(img, verbose=False):
    '''
    Detect background value as the most common value around the edges of
     an image.
    '''
    # Edge values
    edges = np.concatenate([img[:,0], img[0,:], img[:,-1], img[-1,:]])

    # Mode of edge values
    modeValues, counts = mode(edges)

    # Background value
    background = modeValues[0]

    # Report if requested
    if verbose == True: print('Background value: {:f}'.format(background))

    return background



### GENERATE MASK ---
def create_mask(img, maskArgs, refBounds=None, refShape=None, verbose=False):
    '''
    Provide an image array, and one or more files or values to mask.
    Mask values come from the command line as string arguments.
    '''
    # Setup
    M, N = img.shape  # mask size
    mask = np.ones((M, N))  # everything passes preliminary mask


    if maskArgs is not None:
        # Loop through mask arguments
        for maskArg in maskArgs:
            # Determine argument type, i.e., file, value, or rule.
            if os.path.exists(maskArg):
                '''
                Assume map file.
                If a reference boounds (refBounds) and map size (refSize)
                 are provided, resample the mask data set to the reference
                 dimensions.
                '''
                # Open mask image
                mskDS = gdal.Open(maskArg, gdal.GA_ReadOnly)

                # Resample if reference provided
                if refBounds is not None and refShape is not None:
                    assert len(refShape) == 2, \
                        'Reference size must be provided as length-2 list (M, N)'
                    
                    mskDS = gdal_resample(mskDS, bounds=refBounds,
                        M=refShape[0], N=refShape[1],
                        verbose=verbose)

                mskImg = mskDS.GetRasterBand(1).ReadAsArray()

                # Apply to mask
                mask *= mskImg

                # Record inferred argument type
                argType = 'map dataset'


            elif type(maskArg) == str and maskArg.lower() in ['nan']:
                ''' Mask out numpy NaN values. '''
                # Mask NaN values
                mask[np.isnan(img) == 1] = 0

                # Record inferred argument type
                argType = 'numpy nan'


            elif maskArg in ['bg', 'background']:
                ''' Detect and mask background value. '''
                # Detect background value
                background = detect_background(img)

                # Apply to mask
                mask[img == background] = 0

                # Record inferred argument type
                argType = 'background value'


            elif maskArg[0] in ['<', '>']:
                ''' Assume greater than/less than condition. '''
                # Format string to include image reference
                maskArg = 'img {:s}'.format(maskArg)

                # Evaluate
                mask[eval(maskArg)] = 0

                # Record inferred argument type
                argType = 'lt/gt condition'


            else:
                ''' Assume single value. '''
                try:
                    # Try converting to float value
                    maskArg = float(maskArg)

                    # Apply to mask
                    mask[img==maskArg] = 0

                    # Record inferred argument type
                    argType = 'single value'

                except:
                    # Throw hands up \_:(_/
                    print('Mask value {} not recognized.'.format(maskArg))
                    exit()

            # Report if requested
            if verbose == True: print('Mask value {} interpreted as {:s}'.\
                format(maskArg, argType))

    return mask


def mask_dataset(DS, maskArgs, refBounds=None, refShape=None, verbose=False):
    ''' Mask a GDAL data set. '''
    if verbose == True: print('Masking data set')

    # Retrieve image
    img = DS.GetRasterBand(1).ReadAsArray()

    # Create mask
    mask = create_mask(img, maskArgs, refBounds=refBounds, refShape=refShape,
            verbose=verbose)

    return mask


def mask_datasets(datasets, maskArgs, refBounds=None, refShape=None,
        verbose=False):
    ''' Generate masks for multiple GDAL data sets. '''
    if verbose == True:
        print('Generating masks for multiple data sets')

    # Setup
    masks = {}  # empty dictionary

    # Loop through data sets
    for dsName in datasets.keys():
        masks[dsName] = mask_dataset(datasets[dsName], maskArgs,
            refBounds=refBounds, refShape=refShape,
            verbose=verbose)

    return masks


def create_common_mask(datasets, maskArgs, verbose=False):
    '''
    Generate a common (conservative) mask for all data sets.
    Data sets must already be sampled at the same dimensions.
    '''
    if verbose == True:
        print('Creating common mask')

    # Repackage list as dictionary if necessary
    if type(datasets) == list:
        # Name dictionary entries 0 -
        dsNames = range(len(datasets))

        # Repackage as dictionary
        datasets = dict(zip(dsNames, datasets))

    # Check raster sizes are consistent
    M, N = check_dataset_sizes(datasets)

    # Create masks
    masks = mask_datasets(datasets, maskArgs, verbose=verbose)

    # Initial common mask
    commonMask = np.ones((M, N))  # initially pass all values

    # Accumulate mask values
    for mskName in masks.keys():
        # Retrieve individual mask
        mask = masks[mskName]

        # Add to common mask
        commonMask[mask==0] = 0

    return commonMask



### MISCELLANEOUS ---
def find_mask_overlap(primaryMask, secondaryMask):
    ''' Find area where there is data for both masks. '''
    # Setup
    overlap = np.zeros(primaryMask.shape)  # nothing passes at first

    # Find area of overlap
    overlap[(primaryMask > 0) & (secondaryMask > 0)] = 1

    return overlap
