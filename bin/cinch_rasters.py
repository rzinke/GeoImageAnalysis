#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Remove the tilt of one raster relative to another.

FUTURE IMPROVEMENTS
    Add weighting
    Reference to corner coordinates

TESTING STATUS
In development.
'''

### IMPORT MODULES ---
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ImageIO import confirm_outdir, confirm_outname_ext, load_gdal_dataset, save_gdal_dataset
from ImageMasking import create_mask, create_common_mask
from GeoFormatting import DS_to_extent, grid_from_transform
from RasterResampling import match_rasters
from ImageFitting import fit_surface_to_points, design_matrix2d
from ImageViewing import plot_raster, dataset_clip_values


### PARSER ---
Description = '''Remove the tile of one raster image relative to another.'''

Examples = ''''''

def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='refImgName',
        help='Name of reference image.')
    InputArgs.add_argument(dest='secImgName',
        help='Name of image to be tilted.')
    InputArgs.add_argument('-b','--band', dest='bandNbs', nargs=2, default=[1, 1],
        help='Image band number to display. ([1, 1]).')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')
    InputArgs.add_argument('-df','--decimation-factor', dest='decimation', type=int, default=0,
        help='Data set decimation factor, 10^decimation. ([0], 1, 2, ...).')


    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true',
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true',
        help='Plot results.')
    OutputArgs.add_argument('-o','--outname', dest='outname', type=str, default='Clipped',
        help='Untilted map name.')

    return parser

def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### ANCILLARY FUNCTIONS ---
def plot_inputs(cinch, refImgName, secImgName, refBand, secBand, maskArgs):
    ''' Plot resampled input images on same color scale. '''
    # Spawn figure and axes
    fig, axes = plt.subplots(figsize=(9,4), ncols=3)

    # Load data sets
    DSref = load_gdal_dataset(refImgName)
    DSsec = load_gdal_dataset(secImgName)

    # Resample data sets to conservative bounds and resolution
    DSref, DSsec = match_rasters([DSref, DSsec],
                cropping='intersection', resolution='coarse')

    # Retreive resampled extent
    extent = DS_to_extent(DSref)

    # Retreive images
    refImg = DSref.GetRasterBand(refBand).ReadAsArray()
    secImg = DSsec.GetRasterBand(secBand).ReadAsArray()

    # Create common mask
    mask = create_common_mask([DSref, DSsec], maskArgs)

    # Color limits
    vmin, vmax = dataset_clip_values([refImg, secImg], minPct=1, maxPct=99)

    # Plot reference images
    plot_raster(refImg, mask=mask, extent=extent,
        cbarOrient='auto', vmin=vmin, vmax=vmax,
        fig=fig, ax=axes[0])

    # Format reference plot
    axes[0].set_title('Reference image')

    # Plot secondary images
    plot_raster(secImg, mask=mask, extent=extent,
        cbarOrient='auto', vmin=vmin, vmax=vmax,
        fig=fig, ax=axes[1])

    # Format secondary plot
    axes[1].set_title('Secondary image')

    # Plot adjusted image
    plot_raster(cinch.adjustedImg, mask=cinch.mask, extent=cinch.origExtent,
        cbarOrient='auto', vmin=vmin, vmax=vmax,
        fig=fig, ax=axes[2])

    # Format adjusted plot
    axes[2].set_title('Adjusted image')



### UN-TILT CLASS ---
class ImageCinch:
    def __init__(self, refImgName, secImgName, refBand=1, secBand=1,
            maskArgs=None, decimation=0, verbose=False):
        '''
        Cinch one image to another by calculating a ramp representing
         the difference between the two images (secondary minus
         reference), and then subtracting that difference ramp from
         the secondary image.
        '''
        # Parameters
        self.verbose = verbose

        # Load original images
        DSref, DSsec = self.__load_images__(refImgName, secImgName)

        # Format images
        DSref_olap, DSsec_olap, DSmask = \
            self.__format_images_for_comparison__(DSref, DSsec, maskArgs)

        # Solve for difference ramp
        rampCoeffs = self.__solve_for_difference_ramp__(
            DSref_olap, DSsec_olap,
            refBand, secBand,
            DSmask,
            decimation)

        # Remove difference ramp from secondary image
        self.adjustedImg, self.mask = self.__adjust_secondary_image__(
                DSsec, secBand, maskArgs, rampCoeffs)

        # Store secondary image geospatial information
        self.origTnsf = DSsec.GetGeoTransform()
        self.origProj = DSsec.GetProjection()
        self.origExtent = DS_to_extent(DSsec)


    def __load_images__(self, refImgName, secImgName):
        '''
        Load image data sets.
        Resample the images to the same spatial extent and resolution.
        '''
        if self.verbose == True:
            print('Loading image data sets.')

        # Load GDAL data sets
        DSref = load_gdal_dataset(refImgName)
        DSsec = load_gdal_dataset(secImgName)

        return DSref, DSsec


    def __format_images_for_comparison__(self, DSref, DSsec, maskArgs):
        '''
        Format the input images for comparison by:
         (1) checking their overlap
         (2) resampling to common bounds
         (3) computing the common mask
        Returns resampled reference and secondary images, and the common
         mask.
        '''
        # Check image bounds
        self.__check_image_extents__(DSref, DSsec)

        # Resample to same bounds and resolution
        DSref_olap, DSsec_olap = self.__resample_images__(DSref, DSsec)

        # Create and combine masks
        DSmask = self.__generate_mask__(DSref_olap, DSsec_olap, maskArgs)

        return DSref_olap, DSsec_olap, DSmask

    def __check_image_extents__(self, DSref, DSsec):
        '''
        Check that the extent of the secondary is within the extent of
         the reference.
        '''
        if self.verbose == True:
            print('... checking extents')

        # Retreive extents
        refExtent = DS_to_extent(DSref, verbose=self.verbose)
        secExtent = DS_to_extent(DSsec, verbose=self.verbose)

        # Check extents overlap
        # Left
        if refExtent[0] > secExtent[0]:
            print('WARNING! Reference image does not extend to western bound')

        # Right
        if refExtent[1] < secExtent[1]:
            print('WARNING! Reference image does not extend to eastern bound')

        # Bottom
        if refExtent[2] > secExtent[2]:
            print('WARNING! Reference image does not extend to southern bound')

        # Top
        if refExtent[3] < secExtent[3]:
            print('WARNING! Reference image does not extend to northern bound')

    def __resample_images__(self, DSref, DSsec):
        '''
        Resample the input rasters to the same bounds (conservative) and
         resolution (lowest).
        '''
        if self.verbose == True:
            print('... resampling images to same extent and lowest resolution')

        # Resample maps
        DSref_olap, DSsec_olap = match_rasters([DSref, DSsec],
                cropping='intersection', resolution='coarse',
                verbose=self.verbose)

        return DSref_olap, DSsec_olap

    def __generate_mask__(self, DSref_olap, DSsec_olap, maskArgs):
        ''' Create a common mask for both resampled data sets. '''
        if self.verbose == True:
            print('Creating common mask')

        # Create common mask
        DSmask = create_common_mask([DSref_olap, DSsec_olap], maskArgs)

        return DSmask


    def __solve_for_difference_ramp__(self,
            DSref_olap, DSsec_olap,
            refBand, secBand,
            DSmask,
            decimation):
        '''
        Solve for the normal vector for the difference ramp.
         (1) Compute the difference (secondary - reference)
         (2) Fit a linear ramp to the difference
         (3) Subtract the difference ramp from the secondary image
        '''
        if self.verbose == True:
            print('Solving for difference ramp')

        # Retreive images
        secImg = DSsec_olap.GetRasterBand(refBand).ReadAsArray()
        refImg = DSref_olap.GetRasterBand(secBand).ReadAsArray()

        # Difference between two rasters
        diff = secImg - refImg

        # Create X, Y grid
        tnsf = DSref_olap.GetGeoTransform()
        M, N = DSref_olap.RasterYSize, DSref_olap.RasterXSize
        X, Y = grid_from_transform(tnsf, M, N)

        # Remove masked values
        X = X[DSmask == 1]
        Y = Y[DSmask == 1]
        diff = diff[DSmask == 1]

        # Fit ramp to difference
        _, rampCoeffs = fit_surface_to_points(X, Y, diff, degree=1,
                verbose=self.verbose)

        return rampCoeffs


    def __adjust_secondary_image__(self, DSsec, secBand, maskArgs, rampCoeffs):
        '''
        Adjust the secondary image to the reference by removing the
         difference
         (1) Extend difference ramp across all secondary image values
         (2) Subtract ramp
        '''
        if self.verbose == True:
            print('Adjusting secondary image')

        # Retreive secondary image
        img = DSsec.GetRasterBand(secBand).ReadAsArray()

        # Create mask
        mask = create_mask(img, maskArgs)

        # Create X, Y grid
        tnsf = DSsec.GetGeoTransform()
        M, N = DSsec.RasterYSize, DSsec.RasterXSize
        X, Y = grid_from_transform(tnsf, M, N)

        # Create ramp
        G = design_matrix2d(X.flatten(), Y.flatten(), degree=1)
        diffRamp = G.dot(rampCoeffs).reshape(M, N)

        # Subtract difference ramp from secondary image
        adjustedImg = img - diffRamp

        return adjustedImg, mask


    def save(self, outname):
        ''' Save the adjusted image as a GDAL data set. '''
        if self.verbose == True:
            print('Saving adjusted image')

        outname = confirm_outname_ext(inps.outname, ['tif', 'tiff'],
                verbose=inps.verbose)
        save_gdal_dataset(outname, self.adjustedImg, mask=self.mask,
                proj=self.origProj, tnsf=self.origTnsf,
                verbose=inps.verbose)



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Remove ramp
    # Instantiate cinching object
    cinch = ImageCinch(inps.refImgName, inps.secImgName,
            refBand=inps.bandNbs[0], secBand=inps.bandNbs[1],
            maskArgs=inps.maskArgs,
            decimation=inps.decimation,
            verbose=inps.verbose)


    ## Outputs
    # Plotting
    if inps.plot == True:
        plot_inputs(cinch, inps.refImgName, inps.secImgName,
            refBand=inps.bandNbs[0], secBand=inps.bandNbs[1],
            maskArgs=inps.maskArgs)

    # Save data set
    if inps.outname is not None:
        cinch.save(inps.outname)


    plt.show()
