'''
SHORT DESCRIPTION
Image classification.

FUTURE IMPROVEMENTS

TESTING STATUS
Tested.
'''

### IMPORT MODULES ---
import numpy as np



### CLUSTERING ---
class KmeansClustering:
    def __init__(self, data, kClusters, centroids='auto', maxIterations=20,
            verbose=False):
        '''
        Perform a k-means clustering analysis on the data provided.
        INPUTS
            data is an (m x n) numpy array, representing m observations
             over n dimensions
            kClusters is the number of clusters
            centroids is the initial guesses for centroid locations
             If set to default auto, it will pick random data points
             within the data set domain
             Otherwise, a <kClusters> x <nDim> array or nested list
             of user-specified values can be provided
            maxIterations is the maximum number of iterations allowed
        '''
        # Parameters
        self.verbose = verbose

        self.data = data
        self.kClusters = kClusters

        # Parse data set
        self.__parse_dataset__(data)

        # Initialize centroids
        self.__initialize_centroids__(centroids)

        # Home in on clusters by minimizing distances
        self.__find_clusters__(maxIterations)

    def __parse_dataset__(self, data):
        ''' Determine the data set properities. Report if requested. '''
        self.mMeasurements, self.nDim = data.shape

        # Report data properties
        if self.verbose == True:
            print('Dataset')
            print('\tobservations: {:d}'.format(self.mMeasurements))
            print('\tdimensions: {:d}'.format(self.nDim))

    def __initialize_centroids__(self, centroids):
        '''
        Initialize centroid positions.
        If set to auto, then pick random points.
        Ensure one centroid per cluster.
        '''
        if self.verbose == True:
            print('Initializing centroids')

        if centroids == 'auto':
            # Determine data set extrema
            mins = np.min(self.data, axis=0)
            maxs = np.max(self.data, axis=0)

            # Determine stretch factors from extrema
            stretches = maxs - mins

            # Pick random values within extrema
            # One row per cluster, one column per dimension
            self.centroids = np.random.rand(self.kClusters, self.nDim)

            # Scale and shift random numbers to better fit the data domain
            self.centroids = mins + self.centroids*stretches

            # Report if requested
            if self.verbose == True:
                print('\tvalue minima: {}'.format(mins))
                print('\tvalue maxima: {}'.format(maxs))
                print('\tCentroids')
                [print('\t({:d}) {}'.format(k, centroid)) for \
                    k, centroid in enumerate(self.centroids)]

        else:
            if type(centroids) is not np.ndarray:
                assert type(centroids) is list, \
                    ('Centroids must be provided as nested lists with one',
                    'list per cluster')

                # Format lists into numpy array
                centroids = np.array([centroids]).T

            # Ensure number of centroids is consistent with number of clusters
            assert centroids.shape[0] == self.kClusters, \
                'Need same number of centroids as clusters'

            # Ensure number of dimensions is consistent
            assert centroids.shape[1] == self.nDim, \
                'Need same number of centroid parameters as data set dimensions'

            # Checks passed, ascribe to object
            self.centroids = centroids

    def __find_clusters__(self, maxIterations):
        '''
        Home in on the data set clusters.
        The distances from the data points to each centroid will be
         minimized and stored.
        The refinement will take place in two steps:
         1. Calculate the distances from the centroids to the data points
         2. Update the centroid locations based on the mean of each cluster
        If the locations of the centroids do not change, break the loop
        '''
        if self.verbose == True:
            print('Determining cluster centroids')

        # Distances from centroids
        self.dists = np.zeros((self.mMeasurements, self.kClusters))

        # Iteratively determine cluster centers
        for i in range(maxIterations):
            # 1. Calculate the distances from the centroids to the data points
            self.__compute_centroid_data_dists__()

            # 2. Update centroid locations
            convergence = self.__update_centroid_locations__()

            # Finish if no updates were made
            if convergence == True:
                # Report if convergence reached
                if self.verbose == True:
                    print('Convergence reached after {:d} iterations'.\
                        format(i))

                # Break loop
                break

        # Finalize distance measurements
        self.__compute_centroid_data_dists__()

        # Index of final centroid for each data point
        self.centroidIndices = np.argmin(self.dists, axis=1)

    def __compute_centroid_data_dists__(self):
        ''' Compute the distance from the data to each centroid. '''
        for k in range(self.kClusters):
            residuals = self.data - self.centroids[k,:]
            self.dists[:,k] = np.sqrt(np.sum(residuals**2, axis=1))

    def __update_centroid_locations__(self):
        ''' The new centroid location will be the mean of each cluster. '''
        # Restart counter
        updates = 0

        # Determine the index of the smallest distance to each data point
        centroidNdx = np.argmin(self.dists, axis=1)

        # Loop through clusters
        for k in range(self.kClusters):
            # Computer cluster mean
            mean = np.mean(self.data[centroidNdx==k, :], axis=0)

            # Check against previous values
            if not all(self.centroids[k,:]==mean):
                # Increase update counter
                updates += 1

                # Update cluster
                self.centroids[k,:] = mean

        # Report new location if requested
        if (self.verbose == True) and (updates > 0):
            print('\tNew centroid locations')
            [print('\t({:d}) {}'.format(k, centroid)) for \
                k, centroid in enumerate(self.centroids)]

        # Check if updates were made
        if updates == 0:
            return True
        else:
            return False
