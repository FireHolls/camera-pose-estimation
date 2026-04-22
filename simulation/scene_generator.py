import numpy as np

def PointsGenerator(nbPoints, seed, bounds= None, nbPointPlanar = None):
    """
    Function that generates 3D points coordinates from the following inputs:
    - nbPoints      : Integer
                        Number of points to generate
    - seed          : Integer
                        The seed to implement for the random generator to ensure replicability
    - bounds        : np.array (1x6)
                        The lower and upper bounds of x, y and z, respectively
    - nbPointPlanar : Integer <= nbPoints
                        The number of points that are planar 
    Its ouputs are:
    - X             : np.array (3 x nbPoints)
                        Array containing the coordinates of each points in its columns            
    """
    np.random.seed(seed)
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
    else:
        xmin, ymin, zmin = [0, 0, 0]
        xmax, ymax, zmax = [100, 100, 100]

    xs = np.random.uniform(xmin, xmax, size=nbPoints)
    ys = np.random.uniform(ymin, ymax, size=nbPoints)
    zs = np.random.uniform(zmin, zmax, size=nbPoints)

    # If there are points which are planar,
    # make nbPointPlanar of y coordinates equal i.e. planar in XZ
    if nbPointPlanar is not None and (2 <= nbPointPlanar <= nbPoints):
        cst = np.random.uniform(ymin, ymax)
        zs = np.zeros((nbPoints))
        zs[0:nbPointPlanar] = cst

    X = np.zeros((3, nbPoints))
    X[0, :] = xs
    X[1, :] = ys
    X[2, :] = zs
    return X
        
