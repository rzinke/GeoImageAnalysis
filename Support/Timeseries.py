'''
SHORT DESCRIPTION
Compute a simple timeseries of the input data.

FUTURE IMPROVEMENTS
    weighting

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import os
from glob import glob
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

from ImageIO import confirm_outdir, save_gdal_dataset
from ImageMasking import create_mask
from ImageViewing import dataset_clip_values, plot_raster, raster_multiplot



### TIMESERIES CLASS ---
class Timeseries:
    def __init__(self, verbose=False):
        '''
        Compute an (N)SBAS timeseries.
        datePairs are datetime objects in paired lists
        dates are datetime objects representing the times of measurement
         acquisitions
        times are timedeltas since the start of the series
        relative displacements (measurements) are the differences between two
         epochs
        displacements are the cumulative displacements at each date
        '''
        # Set parameters
        self.verbose = verbose

        if self.verbose == True:
            print('Initializing timeseries object')

    def set_date_pairs(self, pairNames):
        '''
        Load datePairs into a list.
        Date pair format is (<secondaryYYYYMMDD>-<referenceYYYYMMDD>).
        '''
        if self.verbose == True:
            print('Loading date pairs')

        # Configure date pairs as datetime objects
        self.__configure_date_pairs__(pairNames)

        # Find individual dates
        self.__individual_dates_from_pairs__()

        # Determine time since start
        self.__dates_to_times__()

    def __configure_date_pairs__(self, pairNames):
        ''' Configure date pairs into datetime objects. '''
        # Empty list
        self.datePairs = []

        # Configure date pairs
        for pairName in pairNames:
            # Split text string at dash
            pairName = pairName.split('-')

            # Convert text string to datetime
            self.datePairs.append([datetime.strptime(dateName, '%Y%m%d') for dateName in pairName])

        # Report if requested
        if self.verbose == True:
            print('... {:d} date pairs set'.format(len(self.datePairs)))

    def __individual_dates_from_pairs__(self):
        ''' Find a list of individual dates from the date pairs. '''
        if self.verbose == True:
            print('Recovering individual dates')

        # Individual dates
        dates = []
        [dates.extend(datePair) for datePair in self.datePairs]

        # Unique dates
        self.dates = list(set(dates))
        self.dates.sort()

        self.nDates = len(self.dates)

        # Report if requested
        if self.verbose == True:
            print('... {:d} unique dates found'.format(self.nDates))

    def __dates_to_times__(self):
        ''' Convert the date strings to timedeltas since the start date. '''
        if self.verbose == True:
            print('Determining time since start')

        # Start date
        self.startDate = self.dates[0]

        # Compute times since start
        times = [date - self.startDate for date in self.dates]

        # Time since start in years
        self.times = [time.days/365.25 for time in times]

    def load_measurements(self, imgList, maskArgs=None, band=1):
        '''
        Load the displacement maps from the list of file names provided.
        Ensure the images are all in the same frame of reference, and are
         listed in the same order as the date pairs.
        '''
        if self.verbose == True:
            print('Loading relative displacement data')

        # Number of images
        self.mMeasurements = len(imgList)
        if self.verbose == True:
            print('... loading {:d} images'.format(self.mMeasurements))

        # Sort lists numerically/alphabetically
        imgList.sort()

        # Load initial data set for reference
        DS0 = gdal.Open(imgList[0], gdal.GA_ReadOnly)

        # Parse initial data set
        imgs = []
        imgs.append(DS0.GetRasterBand(band).ReadAsArray())  # image map
        self.mapProj = DS0.GetProjection()
        self.mapTnsf = DS0.GetGeoTransform()
        self.mapM, self.mapN = DS0.RasterYSize, DS0.RasterXSize

        del DS0  # clear memory

        # Loop through images
        for imgName in imgList[1:]:
            # Load data set
            DS = gdal.Open(imgName, gdal.GA_ReadOnly)
            imgs.append(DS.GetRasterBand(band).ReadAsArray())

            # Check that map bounds and sampling match the original
            assert DS.GetProjection() == self.mapProj, \
                '{:s} projection is not the same as the original'.format(imgName)
            assert DS.GetGeoTransform() == self.mapTnsf, \
                '{:s} geotransform is not the same as the original'.format(imgName)
            assert DS.RasterYSize == self.mapM and DS.RasterXSize == self.mapN, \
                '{:s} size is not the same as the original'.format(imgName)

        # Mask images
        masks = [create_mask(img, maskArgs) for img in imgs]

        # Convert to numpy arrays
        self.imgs = np.array(imgs)  # ( mMaps x mapM x mapN ) array
        self.masks = np.array(masks)  # ( mMaps x mapM x mapN ) array


    def compute_timeseries(self, regularize=False, regularizationWeight=1.0):
        '''
        Compute SBAS-style timeseries.
        At this point, the object should have <mMeasurements> maps of relative
         displacement, and <nDates> epochs. Each map is (<mapM> x <mapN>) in
         size and is registered in the same coordinate system. Each date
         corresponds with a time <times>_i since the start of the series at
         <startDate>.
        Regularization equations assume constant velocity throughout the
         timeseries.
        '''
        if self.verbose == True:
            print('Computing timeseries')

        # Check number of date pairs and number of displacement maps is same
        assert len(self.datePairs) == self.mMeasurements, \
            'Number of date pairs ({:d}) must equal number of images ({:d})'.\
                format(len(self.datePairs), self.mMeasurements)

        # Regularization
        self.regularized = regularize  # regularization applied T/F
        self.regularizationWeight = regularizationWeight  # relative weight

        # Formulate design matrix
        self.__formulate_design_matrix__()

        # Invert for displacements
        self.__invert_network__()

    def __formulate_design_matrix__(self):
        '''
        Create the design matrix.
        Each row in the matrix <G> corresponds to a relative displacement
         measurement. Each column represents a single date in time.
        Without regularization, the design matrix will be <mMeasurements> by
         <nDates> - 1 in size (the reference date is assumed zero displacement
         and is not included in the design matrix).
        If regularization is applied, the matrix is augmented to include terms
         for a linear velocity <V> and an offset constant <C>. The design matrix
         will therefore be (<mMeasurements> + <nDates> - 1) by (<nDates> -1 +2)
         in size.
        '''
        if self.verbose == True:
            print('... formulating design matrix')

        # Empty matrix
        if self.regularized == False:
            mG = self.mMeasurements
            nG = self.nDates - 1
        elif self.regularized == True:
            mG = self.mMeasurements + self.nDates - 1
            nG = self.nDates - 1 + 2
        self.G = np.zeros((mG, nG))

        # Report if requested
        if self.verbose == True:
            print('\tmatrix is {:d} x {:d} ({:d} measurements; {:d} dates)'.\
                format(mG, nG, self.mMeasurements, self.nDates))

        # Indices of dates in design matrix
        self.dateIndices = {}
        for i in range(1, self.nDates):
            self.dateIndices[self.dates[i]] = int(i-1)

        # Loop through measurements to create incidence matrix
        for i in range(self.mMeasurements):
            # Split date pair into dates
            secDate, primDate = self.datePairs[i]

            # Populate design matrix
            self.G[i, self.dateIndices[secDate]] = 1

            if primDate != self.startDate:
                self.G[i, self.dateIndices[primDate]] = -1

        # Apply reguarlization if specified
        if self.regularized == True:
            if self.verbose == True: print('... applying regularization equations')

            # Populate regularization equations
            j = 0  # start counter

            for i in range(self.mMeasurements, self.mMeasurements+self.nDates-1):
                self.G[i, j] = 1  # displacement incidence
                self.G[i,-2] = -self.times[j+1]  # velocity term x time
                self.G[i,-1] = -1  # constant term
                j += 1  # update counter

            # Save design matrix for posterity
            np.savetxt('DesignMatrix.txt', self.G, fmt='%.2f')

    def __invert_network__(self):
        ''' Invert the network to solve for displacements. '''
        if self.verbose == True:
            print('Inverting network ...')

        # Set up displacement array
        self.displacements = np.zeros((self.nDates, self.mapM, self.mapN))

        # Set up velocity and constant arrays if regularization is applied
        if self.regularized == True:
            self.velocity = np.zeros((self.mapM, self.mapN))
            self.constant = np.zeros((self.mapM, self.mapN))

        # Solve for each pixel
        if self.regularized == False:
            # Standard, non-regularized case
            for i in range(self.mapM):
                for j in range(self.mapN):
                    # Solve for displacements
                    displacements = self.__standard_inversion__(i, j)

                    # Write to displacement array
                    self.displacements[1:,i,j] = displacements
        elif self.regularized == True:
            # Regularized inversion
            for i in range(self.mapM):
                for j in range(self.mapN):
                    # Solve for displacements and velocity terms
                    displacements, velocity, constant = \
                        self.__regularized_inversion__(i, j)

                    # Write solutions to arrays
                    self.displacements[1:,i,j] = displacements
                    self.velocity[i,j] = velocity
                    self.constant[i,j] = constant

        # Print when done if requested
        if self.verbose == True: print('Done.')

    def __standard_inversion__(self, i, j):
        ''' Invert for displacements using the standard approach. '''
        # Build weighting matrix
        W = self.__standard_weighting_matrix__(i, j)

        # Retreive data
        data = self.imgs[:,i,j].reshape(-1, 1)

        # Invert for solution
        sln = np.linalg.inv(self.G.T.dot(W).dot(self.G)).dot(self.G.T.dot(W)).dot(data)

        # Format solution
        sln = sln.flatten()

        return sln

    def __standard_weighting_matrix__(self, i, j):
        '''
        Build a weighting matrix for <mMeasurements> observations in
         a standard inversion scenario.
        First, assign weights based on quality metrics (e.g., coherence),
         then, assign null values a very small weight.
        '''
        # Empty matrix
        W = np.zeros((self.mMeasurements, self.mMeasurements))

        # Populate with weight values
        for k in range(self.mMeasurements):
            W[k,k] = 1

        # Weight by coherence
        if hasattr(self, 'cohMaps'):
            for k in range(self.mMeasurements):
                W[k,k] = self.cohMaps[k,i,j]

        # Assign trivial weight to null values
        for k in range(self.mMeasurements):
            if self.masks[k,i,j] == 0:
                W[k,k] = 1E-8

        return W

    def __regularized_inversion__(self, i, j):
        '''
        Invert for displacements and velocity terms using the regularized
         approach.
        '''
        # Build weighting matrix
        W = self.__regularized_weighting_matrix__(i, j)

        # Reshape data
        data = self.imgs[:,i,j].reshape(-1, 1)

        # Extend array for regularization
        data = np.append(data, np.zeros((self.nDates - 1, 1)), 0)

        # Invert for solution
        sln = np.linalg.inv(self.G.T.dot(W).dot(self.G)).dot(self.G.T.dot(W)).dot(data)

        # Format solution
        disp = sln[:self.nDates - 1].flatten()
        v = sln[-2]
        c = sln[-1]

        return disp, v, c

    def __regularized_weighting_matrix__(self, i, j):
        '''
        Build a weighting matrix for <mMeasurements> observations and
         and <nDates> - 1 regularization equations in a regularized
         inversion scenario.
        '''
        # Matrix dimension
        wDim = self.mMeasurements + self.nDates-1

        # Empty matrix
        W = np.zeros((wDim, wDim))

        # Populate with weight values
        for k in range(wDim):
            W[k,k] = 1

        # Weight by coherence
        if hasattr(self, 'cohMaps'):
            for k in range(self.mMeasurements):
                W[k,k] = self.cohMaps[k,i,j]

        # Assign trivial weight to null values
        for k in range(self.mMeasurements):
            if self.masks[k,i,j] == 0:
                W[k,k] = 1E-8

        # Assign weights to regularization equations
        for k in range(self.mMeasurements, wDim):
            W[k,k] = self.regularizationWeight

        return W


    def plot(self):
        ''' Plot the outputs of the network inversion. '''
        if self.verbose == True:
            print('Plotting')

        # Displacements color values
        vmin, vmax = dataset_clip_values([self.displacements[n] for n in range(self.nDates)],
            minPct=1, maxPct=99, masks=None, bounds='outer',
            verbose=self.verbose)

        # Displacements figure
        raster_multiplot([self.displacements[n] for n in range(self.nDates)],
            mrows=3, ncols=4,
            mask=None, extent=None,
            cmap='jet', cbarOrient=None,
            vmin=vmin, vmax=vmax,
            titles=[date.strftime('%Y%m%d') for date in self.dates],
            suptitle='Displacements')

        # Velocity figures
        if self.regularized == True:
            # Plot velocity
            vfig, vax = plot_raster(self.velocity,
                mask=None, extent=None,
                cmap='jet', cbarOrient='auto',
                minPct=0, maxPct=99)
            vax.set_title('Linear velocity')

            # Plot constant
            cfig, cax = plot_raster(self.constant,
                mask=None, extent=None,
                cmap='jet', cbarOrient='auto',
                minPct=0, maxPct=99)
            cax.set_title('Constant shift')


    def save(self, outname):
        ''' Save displacements to multi-band GDAL data set. '''
        if self.verbose == True:
            print('Saving to dataset(s)')

        # Format output names
        confirm_outdir(outname)  # confirm output directory
        dispName = '{:s}_displacements.tif'.format(outname)
        velName = '{:s}_velocity.tif'.format(outname)
        constName = '{:s}_constant.tif'.format(outname)
        datesName = '{:s}_dateList.txt'.format(outname)

        # Save displacements
        save_gdal_dataset(dispName, [self.displacements[i] for i in range(self.nDates)],
            proj=self.mapProj, tnsf=self.mapTnsf, verbose=self.verbose)

        if self.regularized == True:
            # Save velocity
            save_gdal_dataset(velName, self.velocity,
                proj=self.mapProj, tnsf=self.mapTnsf, verbose=self.verbose)

            # Save constant
            save_gdal_dataset(constName, self.constant,
                proj=self.mapProj, tnsf=self.mapTnsf, verbose=self.verbose)

        # Save list of dates in order
        with open(datesName, 'w') as dateFile:
            fmtStr = '{:s}\n'
            for date in self.dates:
                dateFile.write(fmtStr.format(date.strftime('%Y%m%d')))
