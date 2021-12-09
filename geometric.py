from typing import List
from lmfit import Model
import numpy as np
from numpy.linalg import (norm, inv)
from scipy.optimize import (minimize, leastsq)


def radius(c, x, y):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-c[0])**2 + (y-c[1])**2)

def circle_leastsquare(c, xs, ys):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = radius(c, xs, ys)
    return Ri - Ri.mean()

def plane(x,y,A,B,D):
    """ defines a plane equation """
    return (A*x + B*y + D)
  
def project_point_onto_plane(plane_normal: List[float], points: List[List[float]]) -> np.ndarray:
    """ Given a list of points coordinates and a plane normal vector, calculates 
        the coordinates of the projection of points onto the plane """

    # defining plane parameters
    A,B,C,D = plane_normal[0], plane_normal[1], -1, plane_normal[2] # C = -1 by default
    
    projected_points = []
    for point in points:
        x0,y0,z0 = point[0], point[1], point[2]
        # find t that satisfies both plane and line equations
        t = (-A*x0 - B*y0 - C*z0 - D) / (A**2 + B**2 + C**2)
        # extract coordinates from projected point
        x = x0 + A*t
        y = y0 + B*t
        z = z0 + C*t
        # append points into list
        projected_points.append([x,y,z])

    return np.row_stack(projected_points)

def transform(transf_matrix: List[List[float]], points: List[List[float]]) -> np.ndarray:
    """ Transform a list of points with a homogeneous transformation matrix. """

    # initializating list that will contain transformed points
    transformed_points = []
    # iterating over original points
    for point in points:
        point = np.array([[point[0]], [point[1]], [point[2]], [1]])
        # applying transformation with a homogeneos matrix
        transformed_point = transf_matrix @ point
        x = transformed_point[0, 0]
        y = transformed_point[1, 0]
        z = transformed_point[2, 0]
        # appending transformed point to list
        new_point = np.array([x,y,z])
        transformed_points.append(new_point)

    return np.row_stack(transformed_points)

def create_transformation_matrix(Tx: float, Ty: float, Tz: float, Rx: float, Ry: float, Rz: float) -> List[List[float]]:
    """ Creates a homogeneous transformation matrix for point manipulation. """

    # initializating matrix
    transformation_matrix = np.zeros(shape=(4, 4))

    # defining the rotation sub-matrix
    rot_z = np.array([[np.cos(Rz), -np.sin(Rz), 0],[np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(Ry), 0, np.sin(Ry)],[0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
    rotationMatrix = rot_z @ rot_y @ rot_x

    # storing rotation and translation parts on the homogeneos matrix 
    transformation_matrix[:3, :3] = rotationMatrix
    transformation_matrix[3, 3] = 1
    transformation_matrix[:3, 3] = [Tx, Ty, Tz]

    return transformation_matrix


def calculate_euclidian_distance(params: list, *args) -> float:
    """ Function passed to the minimizing method. It calculates the euclidian
        distance between the two lists of points and return it, which in turn 
        will be minimize over the iterations. """

    # extracting parameters and independent variables
    x0 = np.array(args[0])
    x_ref = np.array(args[1])
    (Tx, Ty, Tz, Rx, Ry, Rz) = params

    # initializating array that will contain values that will be minimized
    diff = []
    # iterating over points
    for i in range(np.shape(x0)[0]):
        # calculating rotations
        rot_z = np.array([[np.cos(Rz), -np.sin(Rz), 0],[np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
        rot_y = np.array([[np.cos(Ry), 0, np.sin(Ry)],[0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
        ROT = rot_z @ rot_y @ rot_x
        xr = np.dot(ROT, x0[i])
        # agregating with translations
        xt = xr + np.array([Tx, Ty, Tz])
        # calculating the euclidian distance
        diff.append(((x_ref[i, 0]-xt[0])**2).sum())
        diff.append(((x_ref[i, 1]-xt[1])**2).sum())
        diff.append(((x_ref[i, 2]-xt[2])**2).sum())

    # returning the euclidian distance that will be minimized
    return np.sqrt(np.sum(diff))

def apply_bestfit(points_corresponding: List[float], points_nominal: List[float]):
    """ Computes the deviation from two lists of points by the means of a least-square minimization (best-fit). """
    
    # initializating array with parameters that will be evaluated (R's and T's)
    params = np.zeros(6)

    # applying minimize function to find the best suited transformation parameters
    deviation = minimize(fun=calculate_euclidian_distance, x0=params, args=(points_corresponding, points_nominal),
                         method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

    return deviation

def find_plane_rotations(normal: List[float]) -> List[float]:
    """ Calculates the rotations of a plane by the means of its normal vector. """

    # normalizing normal vector
    normal /= norm(normal)

    # finding a generic perpendicular vector
    perp_normal = np.cross(normal,[normal[1],normal[0]-2,0])

    # normalizing
    perp_normal /= norm(perp_normal)

    # defining 2 point groups to bestfit and find rotations
    origin = [0,0,0]
    pts_nominal = [origin, normal, perp_normal]
    pts_corresponding = [origin, [0,0,1], [1,0,0]]

    # bestfitting points to extract orientations
    deviations = apply_bestfit(pts_corresponding, pts_nominal)
    rotations = deviations[3:]

    return rotations
    
def fit_2D_circle(points: List[List[float]]) -> List[float]:
    """ Calculates the center coodinates and radius of a circle fitted with given points. """

    # first estimation for the fitting
    center_init = np.nanmean(points, axis=0)
    center_estimate = center_init[0], center_init[1]

    # isolating x and y terms
    points = np.row_stack(points)
    xs = points[:, 0]
    ys = points[:, 1]

    # filtering nan values
    xs = list(filter(lambda i: str(i) != 'nan', xs))
    ys = list(filter(lambda i: str(i) != 'nan', ys))

    # applying least square operation to fit the circle
    center, ier = leastsq(circle_leastsquare, center_estimate, args=(xs, ys))

    # calculating circle's radius from the estimated center and the points
    Ri = radius(center, xs, ys)
    R = Ri.mean()

    return [center, R]

def fit_plane_and_extract_normal(xs, ys, zs) -> np.ndarray:
    # fitting a plane
    planeModel = Model(plane, independent_vars=['x', 'y'])
    try:
        result = planeModel.fit(zs, x=xs, y=ys, A=0, B=0, D=0)
    except ValueError:
        # case in which there are rows/point names with no values
        xs = list(filter(lambda i: str(i) != 'nan', xs))
        ys = list(filter(lambda i: str(i) != 'nan', ys))
        zs = list(filter(lambda i: str(i) != 'nan', zs))
        result = planeModel.fit(zs, x=xs, y=ys, A=0, B=0, D=0)
    a, b, d = result.best_values['A'], result.best_values['B'], result.best_values['D']
    plane_normal = np.array([a,b,d])

    return plane_normal

def create_new_frame_and_transform_points(plane_normal, origin_point, projPts):
    # defining a transformation between the current frame and the one defined by the plane's normal
    rot = find_plane_rotations(plane_normal)
    # changuing point's frame/base
    transf_matrix = create_transformation_matrix(origin_point[0], origin_point[1], origin_point[2], rot[0], rot[1], rot[2])
    projPts = transform(inv(transf_matrix), projPts)
    return projPts

def fit_circle_with_points_projected_into_plane(points):
    # isolating x, y and z terms
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    # fitting a plane
    plane_normal = fit_plane_and_extract_normal(xs, ys, zs)
    # projecting points onto plane 
    projPts = project_point_onto_plane(plane_normal, points)
    # create new frame with directions determined by the plane normal and with origin somewhere in the plane
    projPts_transformed = create_new_frame_and_transform_points(plane_normal, projPts[0], projPts)
    # fitting 2D circle
    center_x, center_y, _ = fit_2D_circle(projPts_transformed)
    return [center_x, center_y, projPts_transformed]
