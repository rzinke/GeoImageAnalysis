'''
SHORT DESCRIPTION
Image profile support.

FUTURE IMPROVEMENTS

TESTING STATUS
In development.

POLYLINE NAMING CONVENTION
                    profile
                       o
                       | }-spacing
                       o
                       |
                       o
                       |
                 __.___.___.__. polyline
                /      |
               .       o
              /        |
    width    /         o
    |   |   .          |
 ___.___.__/           o
 \ /       
 offset

PROFILE NAMING CONVENTION
   _____
  |     |
  |     | \
  | -w- | length
  |     | /
  |     |
--x--o--x- anchor
  |w2 w2|
  |     |
'''

### IMPORT MODULES ---
import numpy as np
from scipy.interpolate import interp1d


### VECTOR MATH ---
def rotation_matrix(theta):
    '''
    Definition of a standard rotation matrix.
    CCW is positive.
    Theta given in radians.
    '''
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    return R


def determine_pointing_vector(pxStart, pyStart, pxEnd, pyEnd, verbose=False):
    '''
    Find the pointing vector between the two points and the original vector
     length.
    '''
    # Find pointing vector
    p = np.array([pxEnd-pxStart, pyEnd-pyStart])
    pLen = np.linalg.norm(p)
    p = p/pLen

    # Report if requested
    if verbose == True:
        print('unit vector: {:.3f}, {:.3f}'.format(*p))
        print('vector length: {:f}'.format(pLen))

    return p, pLen


def pointing_vector_to_angle(px, py, verbose=False):
    '''
    Convert pointing vector px, py to angle.
    Theta returned in radians.
    '''
    theta = np.arctan2(py, px)

    return theta


def rotate_coordinates(X, Y, theta, verbose=False):
    ''' Rotate X, Y coordinates by the angle theta. '''
    # Setup
    M, N = X.shape
    MN = M*N

    R = rotation_matrix(theta)

    # Reshape coordinate points into 2 x MN array for rotation
    C = np.vstack([X.reshape(1, MN),
                   Y.reshape(1, MN)])
    del X, Y  # clear memory

    # Rotate coordinates
    C = R.dot(C)

    # Reshape coordinate matrices
    X = C[0,:].reshape(M,N)
    Y = C[1,:].reshape(M,N)
    del C  # clear memory

    return X, Y



### DISTANCE ALONG PROFILE ---
def polyline_length(x, y):
    '''  Calculate the distance along a polyline. '''
    # Parameters
    assert len(x) == len(y)
    N = len(x)  # number of data points
    distances = np.zeros(N)  # empty array of distances

    # Loop through each point
    for i in range(N-1):
        # Distance components
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]

        # Euclidean distance
        d = np.sqrt(dx**2 + dy**2)

        distances[i+1] = distances[i] + d  # start with zero, add cumulative

    return distances


def points_along_polyline(lineX, lineY, width, offset=0, verbose=False):
    '''
    Compute the x,y coordinates of points along a polyline at the specified
     spacing (width).

    INPUTS
        lineX, lineY are the coordinates of the polyline vertices
        width is the profile width
        offset is the distance between the first vertex and the first point

    OUTPUTS
        qx, qy are the profile points
    '''
    # Compute the distance array along the profile
    l = polyline_length(lineX, lineY)

    # Array of query points (function of distance)
    q = np.arange(offset, l[-1], width)

    # Interpolate along distance axis
    I = interp1d(l, np.column_stack([lineX, lineY]), axis=0)  # interp function

    # Solve for query point coordinates
    coords = I(q)
    qx = coords[:,0]  # x-coordinates
    qy = coords[:,1]  # y-coordinates

    return qx, qy


def find_profile_anchors(lineX, lineY, width, verbose=False):
    ''' Find the anchor points of profiles along a polyline. '''
    # Find start anchors (0 offset)
    startAnchors = points_along_polyline(lineX, lineY, width, verbose=verbose)
    startAnchors = np.column_stack(startAnchors)

    # Find end anchors (w offset)
    endAnchors = points_along_polyline(lineX, lineY, width, offset=width, verbose=verbose)
    endAnchors = np.column_stack(endAnchors)

    # Clip start anchors to same number as end anchors
    startAnchors = startAnchors[:endAnchors.shape[0],:]

    return startAnchors, endAnchors


def find_profile_geometries(lineX, lineY, profWidth, profLen, verbose=False):
    ''' Determine the coordinates of profile vertices along a polyline. '''
    # Setup
    profCoords = {}

    # Find the profile anchors along the polyline
    startAnchors, endAnchors = find_profile_anchors(lineX, lineY, profWidth, verbose=verbose)

    # Create profGeom objects to describe the profile coordinates
    nAnchors = len(startAnchors)
    profGeoms = []

    for i in range(nAnchors):
        profGeom = profile_geometry(verbose=True)
        profGeom.from_anchors(startAnchors[i], endAnchors[i], profLen)
        profGeoms.append(profGeom)

    return profGeoms


class profile_geometry:
    def __init__(self, verbose=False):
        '''
        Compute and store the coordinates and properties composing a profile.
        '''
        self.verbose = verbose

    def from_endpoints(self, profStart, profEnd, profWidth):
        '''
        Provide the starting and ending points of the profile.
        Automatically calculate the profile geometry.
        Values given in map units.
        '''
        self.profStart = profStart
        self.profEnd = profEnd
        self.profWidth = profWidth

        # Determine vectors
        self.__determine_vectors_from_start_end__()

        # Determine profile corners
        self.__determine_corners__()

        # Check components
        self.__check_components__()

    def from_anchors(self, startAnchor, endAnchor, profLen):
        '''
        Provide the starting and ending anchor points along the polyline.
        Automatically calculate the profile geometry.
        Values given in map units.
        '''
        self.startAnchor = startAnchor
        self.endAnchor = endAnchor
        self.profLen = profLen

        # Determine mid points
        self.__find_anchor_midpoint__()

        # Determine vectors
        self.__determine_vectors_from_anchors__()

        # Determine profile start and end
        self.__determine_start_end__()

        # Determine profile corners
        self.__determine_corners__()

        # Check components
        self.__check_components__()

    def __find_anchor_midpoint__(self):
        ''' Find the midpoint of the profile. '''
        self.midAnchor = (self.endAnchor - self.startAnchor)/2 + self.startAnchor

    def __determine_vectors_from_start_end__(self):
        '''
        Vectors describing the width and length directions based on start and
         end points.
        '''
        self.lenVector, self.profLen = determine_pointing_vector(*self.profStart, *self.profEnd)
        self.widthVector = np.array([-self.lenVector[1], self.lenVector[0]])

    def __determine_vectors_from_anchors__(self):
        '''
        Vectors describing the width and length directions based on anchor points.
        '''
        self.widthVector, self.profWidth = determine_pointing_vector(*self.startAnchor, *self.endAnchor)
        self.lenVector = np.array([-self.widthVector[1], self.widthVector[0]])

    def __determine_start_end__(self):
        ''' Determine the start and end points of a profile. '''
        halfLen = self.profLen/2
        self.profStart =  halfLen*self.lenVector+self.midAnchor
        self.profEnd   = -halfLen*self.lenVector+self.midAnchor

    def __determine_corners__(self):
        ''' Determine profile corners. '''
        halfWidth = self.profWidth/2
        corner1 =  halfWidth*self.widthVector+self.profStart
        corner2 =  halfWidth*self.widthVector+self.profEnd
        corner3 = -halfWidth*self.widthVector+self.profEnd
        corner4 = -halfWidth*self.widthVector+self.profStart

        self.corners = np.vstack([corner1, corner2, corner3, corner4, corner1])

    def __check_components__(self):
        ''' Check that the object has the necessary components. '''
        assert hasattr(self, 'profStart'), 'Starting point required'
        assert hasattr(self, 'profEnd'), 'End point required'
        assert hasattr(self, 'profWidth'), 'Width required'
        assert hasattr(self, 'profLen'), 'Length required'
        assert hasattr(self, 'corners'), 'Corners must be computed'

        # Report if requested
        if self.verbose == True:
            print('Profile:')
            print('\tstart: {:f} {:f}'.format(*self.profStart))
            print('\tend: {:f} {:f}'.format(*self.profEnd))
            print('\twidth: {:f}'.format(self.profWidth))
            print('\tlength: {:f}'.format(self.profLen))



### PROFILING ---
def extract_profile(img, pxStart, pyStart, pxEnd, pyEnd, pxWidth, mask=None, verbose=False):
    '''
    Extract a profile from an image.
    INPUTS
        pxStart and pxEnd are the start and end points of the profile
        profWidth is the width of the profile

        * All units are given in pixel values, not geographic coodinates.
    '''
    if verbose == True: print('Extracting profile...')

    # Parameters
    M, N = img.shape
    MN = M*N

    # Build grid
    x = np.arange(N)
    y = np.arange(M)

    X, Y = np.meshgrid(x, y)

    # Format profile grid
    X, Y, pxLen = format_profile_grid(X, Y, pxStart, pyStart, pxEnd, pyEnd, verbose=verbose)

    # Extract valid pixels
    profDist, profPts = extract_profile_values(img, X, Y, pxLen, pxWidth, mask=mask, verbose=verbose)

    return profDist, profPts

def format_profile_grid(X, Y, pxStart, pyStart, pxEnd, pyEnd, verbose=False):
    '''
    Format the pixel grid such that X, Y are centered and rotated such that
     X increases with distance along the profile length, and Y increases or
     decreases with profile width.
    '''
    if verbose == True: print('... formatting grid')
    # Recenter at starting pixel
    X = X - pxStart
    Y = Y - pyStart

    # Determine direction unit vector p
    p, pxLen = determine_pointing_vector(pxStart, pyStart, pxEnd, pyEnd, verbose=verbose)

    # Determine rotation angle
    theta = pointing_vector_to_angle(p[0], p[1], verbose=verbose)

    # Rotate coordinates
    X, Y = rotate_coordinates(X, Y, -theta)

    return X, Y, pxLen

def extract_profile_values(img, X, Y, pxLen, pxWidth, mask=None, verbose=False):
    '''
    Extract the valid pixels from an image given X and Y grid values that are
     already rotated into profile-centric coordinate system (e.g., using the
     "extract profile" function.)

    INPUTS
        img is the image from while the profile is to be extracted
        X, Y are the image coordinate arrays, centered and rotated such that 
         X increases with distance along the profile length, and Y increases
         or decreases with profile width.

        * All units are given in pixel values, not geographic coordinates.
    '''
    if verbose == True: print('... extracting pixels')

    # Parameters
    w2 = int(pxWidth/2)

    # Extract valid pixels
    validPts = np.ones(img.shape)
    validPts[X>pxLen] = 0
    validPts[X<0] = 0
    validPts[Y<-w2] = 0
    validPts[Y>w2] = 0

    if mask is not None:
      validPts[mask == 0] = 0

    # Extract valid points
    profDist = X[validPts == 1].flatten()  # point distances
    profPts = img[validPts == 1].flatten()  # valid points
    del validPts  # clear memory

    return profDist, profPts



### BINNING ---
def profile_binning(profDist, profPts, binWidth=None, binSpacing=None):
    '''
    Find the bin values of unevenly sampled and redundant data along a profile.
    This is essentially equivalent to a moving median filter.
    '''
    # Parameters
    if binSpacing is None:
        sortedDists = np.sort(profDist)  # sort profDist smallest to largest
        binSpacing = 4*np.mean(np.diff(sortedDists))  # mean distance between points
    if binWidth is None:
        binWidth = binSpacing*2  # twice the bin spacing

    # Setup
    w2 = binWidth/2  # half-width of bins
    x = np.arange(w2, profDist.max()-w2, binSpacing)
    nBins = len(x)  # number of bins
    binStarts = x - w2  # starting point of each bin
    binEnds = x + w2  # ending point of each bin
    y = np.empty(nBins)
    y[:] = np.nan

    # Find points in each bin
    for n in range(nBins):
        binPts = profPts[(profDist >= binStarts[n]) & (profDist < binEnds[n])]
        if len(binPts) > 0:
            y[n] = np.nanmedian(binPts)

    return x, y
