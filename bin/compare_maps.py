#!/usr/bin/env python3
'''
SHORT DESCRIPTION
Compare two rasters.

FUTURE IMPROVEMENTS
    * Multiple analysis inputs
    * K-means clustering

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltColors

from ImageIO import load_gdal_datasets, confirm_outname_ext
from RasterResampling import match_rasters
from ImageMasking import create_common_mask
from GeoFormatting import transform_to_extent
from ImageViewing import plot_raster, histogram2d, kde2d



### PARSER ---
Description = '''View and compare two raster data sets.'''

Examples = ''''''


def createParser():
    parser = argparse.ArgumentParser(description=Description,
        formatter_class=argparse.RawTextHelpFormatter, epilog=Examples)

    InputArgs = parser.add_argument_group('INPUTS')
    InputArgs.add_argument(dest='refImgName',
        help='Name of reference image.')
    InputArgs.add_argument(dest='secImgName',
        help='Name of comparison image.')
    InputArgs.add_argument('-m','--mask', dest='maskArgs', nargs='+', type=str, default=None,
        help='Arguments for masking values/maps. ([None]).')
    InputArgs.add_argument('-b','--band', dest='bandNb', type=int, default=1,
        help='Image band number to display.')

    MapArgs = parser.add_argument_group('MAP DISPLAY PARAMS')
    MapArgs.add_argument('-c','--cmap', dest='cmap', type=str, default='viridis',
        help='Colormap. [viridis].')
    MapArgs.add_argument('-co','--colorbar-orientation', dest='cbarOrient', type=str, default='horizontal',
        help='Colorbar orientation ([horizontal], vertical).')
    MapArgs.add_argument('-minPct','--min-percent', dest='minPct', type=float, default=None,
        help='Minimum percent clip value ([None]).')
    MapArgs.add_argument('-maxPct','--max-percent', dest='maxPct', type=float, default=None,
        help='Maximum percent clip value ([None]).')

    DataPlotArgs = parser.add_argument_group('DATA PLOT PARAMS')
    DataPlotArgs.add_argument('-pt','--plot-type', dest='plotType', type=str,
        choices=['points', 'hexbin', 'hist', 'kde', 'contour', 'contourf'], default='points',
        help='Method of displaying data. [points]')
    DataPlotArgs.add_argument('-ds','--downsampling-factor', dest='dsFactor', type=str, default='auto',
        help='Downsampling factor: 2^ds ([\'auto\'], 1, 2, 3, ...)')
    DataPlotArgs.add_argument('--nbins', dest='nbins', type=int, default=30,
        help='Number of bins on a side for 2D histogram.')
    DataPlotArgs.add_argument('--log-density', dest='logDensity', action='store_true',
        help='Normalize heatmap colors to log10.')

    AnalysisArgs = parser.add_argument_group('ANALYSIS')
    AnalysisArgs.add_argument('-a','--analysis-type', dest='analysisType', type=str,
        choices=[None, 'polyfit', 'pca'], default=None,
        help='Analysis type. [None]')
    AnalysisArgs.add_argument('--fit-order', dest='fitOrder', type=int, default=1,
        help='Polyfit order ([1], 2, 3, ...).')

    OutputArgs = parser.add_argument_group('OUTPUTS')
    OutputArgs.add_argument('-v','--verbose', dest='verbose', action='store_true', 
        help='Verbose mode.')
    OutputArgs.add_argument('-p','--plot', dest='plot', action='store_true', 
        help='Plot outputs.')
    OutputArgs.add_argument('--data-file', dest='dataFile', type=str, default=None,
        help='File to which to write data comparison (column 1 reference image pixel values / column 2 secondary pixel values). [None].')

    return parser


def cmdParser(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)



### COMPARISON ---
class MapComparison:
    def __init__(self, refImgName, secImgName, maskArgs=None, verbose=False):
        ''' Plot and compare two images. '''
        # Parameters
        self.verbose = verbose

        # Data set names (basename with extension removed)
        self.refName = '.'.join(os.path.basename(refImgName).split('.')[:-1])
        self.secName = '.'.join(os.path.basename(secImgName).split('.')[:-1])


        ## Loading and formatting
        # Load images
        self.__load_images__(refImgName, secImgName, maskArgs)


    ## Loading
    def __load_images__(self, refImgName, secImgName, maskArgs=None):
        ''' Load images and resample to common extent. '''
        # Load data sets
        DSs = load_gdal_datasets([refImgName, secImgName], verbose=self.verbose)

        # Match spatial extent
        DSs = match_rasters(DSs, cropping='intersection', resolution='coarse',
                verbose=self.verbose)

        # Create mask
        self.mask = create_common_mask(DSs, maskArgs, verbose=self.verbose)

        # Record images
        DSs = list(DSs.values())
        self.refImg = DSs[0].GetRasterBand(1).ReadAsArray()
        self.secImg = DSs[1].GetRasterBand(1).ReadAsArray()

        # Remove NaNs
        self.refImg[np.isnan(self.refImg)==1] = 0
        self.secImg[np.isnan(self.secImg)==1] = 0

        # Record data set geographic properties
        self.M, self.N = DSs[0].RasterYSize, DSs[0].RasterXSize
        self.tnsf = DSs[0].GetGeoTransform()
        self.proj = DSs[0].GetProjection()

        # Geographic transform
        self.extent = transform_to_extent(self.tnsf, self.M, self.N,
                verbose=self.verbose)

        # Clear memory
        del DSs


    ## Analysis
    def analysis(self, analysisType=None, fitOrder=1):
        ''' Analyze the point-by-point comparison. '''
        if self.verbose == True: print('Analyzing data points.')

        # Record parameters
        self.analyses = []  # record analysis types
        self.fitOrder = fitOrder  # polyline fit order

        # Mask data points
        self.refData = np.ma.array(self.refImg, mask=(self.mask==0))
        self.secData = np.ma.array(self.secImg, mask=(self.mask==0))

        # Compress and flatten arrays
        self.refData = self.refData.compressed().flatten()
        self.secData = self.secData.compressed().flatten()

        # Array lengths of valid pixels
        assert len(self.refData) == len(self.secData), \
            'Reference and secondary data sets are not the same size ({:d} vs {:d})'.\
            format(len(self.refData), len(self.secData))
        self.nPts = len(self.refData)

        # Report difference
        if self.verbose == True:
            difference = self.secData - self.refData
            medianDiff = np.median(difference)
            meanDiff = np.mean(difference)
            meanAbsDiff = np.mean(np.abs(difference))
            RMS = np.sqrt(np.mean(difference**2))

            print('Differences (secondary - reference)')
            print('\tMedian difference:   {:f}'.format(medianDiff))
            print('\tMean difference:     {:f}'.format(meanDiff))
            print('\tMean abs difference: {:f}'.format(meanAbsDiff))
            print('\tRMS difference:      {:f}'.format(RMS))

        # Analyze
        if analysisType is not None:
            if analysisType.lower() in ['polyfit']:
                # Record analysis type
                self.analyses.append('polyfit')

                # Polyline fit analysis
                self.__polyfit__()

            elif analysisType.lower() in ['pca']:
                # Record analysis type
                self.analyses.append('pca')

                # Principal component analysis
                self.__pca_fit__()

            else:
                print('Analysis type not recognized.')
                exit()
        else:
            if self.verbose == True: print('No analysis type specified.')

    def __polyfit__(self):
        ''' Fit the data trend with an nth-order fit. '''
        if self.verbose == True:
            print('Analyzing with {:d}-order polyline fit.'.format(self.fitOrder))

        # Create design matrix
        G = self.__create_design_matrix__(self.refData, self.fitOrder)

        # Invert for solution
        self.B = np.linalg.inv(np.dot(G.T, G)).dot(G.T).dot(self.secData)

        # Fit solution
        yhat = G.dot(self.B)

        # Compute residuals
        res = yhat - self.secData

        # Root mean square of residuals
        RMS = np.sqrt(np.mean(res**2))

        # Report if requested
        if self.verbose == True:
            print('Beta')
            [print('\tx**{:d}: {:f}'.format(i, self.B[i])) for i in range(self.fitOrder+1)]

            print('RMS(resid): {:f}'.format(RMS))

    def __pca_fit__(self):
        ''' Solve for principal components. '''
        if self.verbose == True: print('Analyzing using PCA.')

        # Stack data into column matrix
        data = np.column_stack([self.refData, self.secData])

        # Remove centroid
        dataMean = np.mean(data, axis=0)
        data = data - dataMean
        self.refMean, self.secMean = dataMean  # record for posterity

        # Covariance matrix
        C = np.dot(data.T, data)/(data.shape[0]-1)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eig(C)
        sortNdx = np.argsort(eigvals)[::-1]  # reorder
        eigvals = eigvals[sortNdx]
        eigvecs = eigvecs[:, sortNdx]

        # Standard devations
        eigstds = np.sqrt(eigvals)

        # Construct principal components
        self.PC1 = eigstds[0]*eigvecs[:,0]
        self.PC2 = eigstds[1]*eigvecs[:,1]

        # Relationship between eigenvectors
        eigslope = eigvecs[1,0]/eigvecs[0,0]

        # Report if requested
        if self.verbose == True:
            print('Eigenvalues:\n\t{:f}\n\t{:f}'.format(*eigvals))
            print('Eigenvectors:')
            [print('\t{:f}, {:f}'.format(*eigvec)) for eigvec in eigvecs]
            print('Relationship: {:.3f}'.format(eigslope))

    def __create_design_matrix__(self, x, fitOrder):
        ''' Create a design matrix with columns increasing in order. '''
        # Parameters
        n = len(x)

        # Start with matrix of ones
        G = np.ones((n, fitOrder+1))

        for i in range(fitOrder+1):
            G[:,i] = x**i

        return G


    ## Plotting
    def plot_maps(self, cmap='viridis', cbarOrient='horizontal', minPct=None, maxPct=None):
        ''' Plot inputs and differnce map. '''
        # Determine figure setup
        if self.M < self.N:
            nrows = 3
            ncols = 1
        else:
            nrows = 1
            ncols = 3

        # Spawn figure
        fig, [axRef, axSec, axDiff] = plt.subplots(nrows=nrows, ncols=ncols)

        # Plot reference figure
        plot_raster(self.refImg, mask=self.mask, extent=self.extent,
            cmap=cmap, cbarOrient=cbarOrient,
            minPct=minPct, maxPct=maxPct,
            fig=fig, ax=axRef)

        # Plot secondary figure
        plot_raster(self.secImg, mask=self.mask, extent=self.extent,
            cmap=cmap, cbarOrient=cbarOrient,
            minPct=minPct, maxPct=maxPct,
            fig=fig, ax=axSec)

        # Plot difference map
        plot_raster(self.secImg-self.refImg, mask=self.mask, extent=self.extent,
            cmap='cividis', cbarOrient=cbarOrient,
            minPct=minPct, maxPct=maxPct,
            fig=fig, ax=axDiff)

        # Format figure
        axRef.set_title('Reference image')
        axSec.set_title('Secondary image')
        axDiff.set_title('Difference image\n(sec - ref)')
        fig.tight_layout()

    def plot_data(self, plotType='pts', dsFactor='auto', nbins=30, logDensity=False):
        '''
        Plot pixel comparison as points, etc.
        Wrapper for data plotting functions.
        '''
        if self.verbose == True:
            print('Plotting data with type: {:s}'.format(plotType.upper()))

        # Parameters
        self.dsFactor = dsFactor  # point downsampling factor
        self.nbins = nbins  # nb bins on a side for 2D histogram
        self.logDensity = logDensity  # normalize colors to log10

        # Spawn plot and axis
        self.dataFig, self.axData = plt.subplots()

        # Plot figure by type
        if plotType in ['pts', 'points', 'scatter']:
            self.__plot_pts__()

        elif plotType in ['hex', 'hexbin']:
            self.__plot_hex__()

        elif plotType in ['hist', 'hist2d', 'histogram', 'histogram2d']:
            self.__plot_hist__()

        elif plotType in ['kde']:
            self.__plot_kde__()

        elif plotType in ['contour', 'contours']:
            self.__plot_contours__()

        elif plotType in ['contourf']:
            self.__plot_contourf__()

        else:
            print('Data figure type not recognized.')
            exit()

        # Format plot
        self.axData.set_xlabel('reference data values')
        self.axData.set_ylabel('secondary data values')

    def __plot_pts__(self):
        ''' Plot data as scatter points. '''
        # Determine downsampling
        self.__determine_downsampling__(self.dsFactor)

        # Plot data points
        self.axData.scatter(self.refData[::self.ds], self.secData[::self.ds],
            color='gray')

    def __determine_downsampling__(self, dsFactor):
        '''
        Determine the appropriate downsampling factor.
        If the factor is not explicitly specified, determine what it should be
         based on the image sizes.
        '''
        if dsFactor == 'auto':
            maxPts = 100000  # max number of data points to plot
            self.ds = np.ceil(self.nPts/maxPts)
        else:
            self.ds = 2**float(dsFactor)
        self.ds = int(self.ds)  # convert to integer

        # Report if requested
        if self.verbose == True:
            print('... skipping every {:d} points'.format(self.ds))

    def __plot_hex__(self):
        ''' Plot using the hexbin scheme. '''
        # Color normalization
        if self.logDensity == True:
            colorNorm = pltColors.LogNorm()
        else:
            colorNorm = None

        # Plot hexbins
        cax = self.axData.hexbin(self.refData, self.secData,
            cmap='viridis', norm=colorNorm)

        # Format colorbar
        self.dataFig.colorbar(cax, ax=self.axData, orientation='horizontal')

    def __plot_hist__(self):
        ''' Plot using 2D histogram scheme. '''
        # Plot histogram
        histogram2d(self.refData, self.secData,
            nbins=self.nbins,
            logDensity=self.logDensity,
            fig=self.dataFig, ax=self.axData)

    def __plot_kde__(self):
        ''' Plot using 2D kernel density estimate scheme. '''
        # Plot KDE
        kde2d(self.refData, self.secData,
            nbins=self.nbins,
            logDensity=self.logDensity,
            fig=self.dataFig, ax=self.axData)

    def __plot_contours__(self):
        ''' Plot contours on top of point measurements. '''
        # Plot points
        self.__plot_pts__()

        # Plot KDE
        kde2d(self.refData, self.secData,
            plotType='contour',
            nbins=self.nbins,
            logDensity=self.logDensity,
            fig=self.dataFig, ax=self.axData)

    def __plot_contourf__(self):
        ''' Plot using 2D kernel density estimate as filled contours. '''
        # Plot KDE
        kde2d(self.refData, self.secData,
            plotType='contourf',
            nbins=self.nbins,
            logDensity=self.logDensity,
            fig=self.dataFig, ax=self.axData)

    def plot_analysis(self):
        '''
        Plot the results of the analysis.
        Wrapper for analysis plotting functions.
        '''
        # Plot based on analysis type
        if 'polyfit' in self.analyses:
            self.__plot_polyline__()
        if 'pca' in self.analyses:
            self.__plot_pca__()

        # Format plot
        if len(self.analyses) > 0:
            self.axData.legend()

    def __plot_polyline__(self):
        ''' Plot polyline fit. '''
        # X-values
        x = np.linspace(self.refData.min(), self.refData.max(), 100)

        # Create fit function
        G = self.__create_design_matrix__(x, self.fitOrder)
        y = G.dot(self.B)

        # Plot fit
        self.axData.plot(x, y, 'k--', label='polyfit')

    def __plot_pca__(self):
        ''' Plot principal components. '''
        # Plot data centroid
        self.axData.plot(self.refMean, self.secMean, 'ko')

        # Plot eigenbasis
        scale = 3  # standard deviations
        self.axData.plot([-scale*self.PC1[0]+self.refMean, scale*self.PC1[0]+self.refMean],\
            [-scale*self.PC1[1]+self.secMean, scale*self.PC1[1]+self.secMean], 'k--',
            label='PC1')
        self.axData.plot([-scale*self.PC2[0]+self.refMean, scale*self.PC2[0]+self.refMean],\
            [-scale*self.PC2[1]+self.secMean, scale*self.PC2[1]+self.secMean], 'k--',
            label='PC2')


    ## Saving
    def save_to_file(self, datafname):
        '''
        Save data columns to a text file, excluding masked data.
        Column 1: reference image pixel values
        Column 2: secondary image pixel values
        '''
        # Check file extension
        datafname = confirm_outname_ext(datafname, ['txt', 'dat', 'csv'])

        # Create text file
        with open(datafname, 'w') as datafile:
            # Write header
            header = '# {:s} {:s}\n'.format(self.refName, self.secName)
            datafile.write(header)

            # Save data
            np.savetxt(datafile, np.column_stack([self.refData, self.secData]))

        # Report if requested
        if self.verbose == True:
            print('Saved unmasked data columns to {:s}'.format(datafname))



### MAIN ---
if __name__ == '__main__':
    ## Inputs
    # Gather arguments
    inps = cmdParser()


    ## Comparison
    compare = MapComparison(inps.refImgName, inps.secImgName,
        maskArgs=inps.maskArgs,
        verbose=inps.verbose)

    # Comparison analysis
    compare.analysis(analysisType=inps.analysisType, fitOrder=inps.fitOrder)


    ## Plotting
    if inps.plot == True:
        # Plot maps
        compare.plot_maps(cmap=inps.cmap, cbarOrient=inps.cbarOrient,
            minPct=inps.minPct, maxPct=inps.maxPct)

        # Plot data
        compare.plot_data(plotType=inps.plotType, dsFactor=inps.dsFactor, nbins=inps.nbins, logDensity=inps.logDensity)

        # Plot analysis
        compare.plot_analysis()

        # Save data columns to text file
        compare.save_to_file(inps.dataFile)

        plt.show()
