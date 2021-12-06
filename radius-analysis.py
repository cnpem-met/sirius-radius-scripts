# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:07:56 2021

@author: rodrigo.neto
"""

from lmfit import Model
import numpy as np
from numpy.linalg import (norm, inv)
from scipy.optimize import (minimize, leastsq)
import os
import pandas as pd

def readCSV(fileDir):

    # create header
    header = ['Point', 'x', 'y', 'z']

    # reading csv file with automatic detection of its delimeter
    data = pd.read_csv(fileDir, sep=None, engine='python', names=header)

    # checking if it has a bult-in header and droping it
    if (type(data.iloc[0, 1]) is str):
        data = data.drop([0])

    # sorting and indexing df
    data = data.sort_values(by=header[0])
    data.reset_index(drop=True, inplace=True)
    data = data.set_index(header[0])

    return data

def readAllTxtInPath(path, mode):
    rawData, dateRef = [], []
    expectedPtNames = {'external': ['RC1_Exter_P1', 'RC2_Exter_P1', 'RC3_Exter_P1', 'RC4_Exter_P1', 'RC5_Exter_P1'],
                        'internal': ['RC1', 'RC2', 'RC3', 'RC4', 'RC5'],
                        'magnets': ['RC1_Ima', 'RC2_Ima', 'RC3_Ima', 'RC4_Ima', 'RC5_Ima']}
    # listing all files in path
    files = os.listdir(path)
    files_full_path = [os.path.join(path, file) for file in files]
    # ordering list according to modification time
    sorted_files = sorted(files_full_path, key=lambda f: os.stat(f).st_ctime)
    # iterating over all files in the path
    for file in sorted_files:
        # considering only .txt extension
        if file.endswith('.txt'):
            # read file and store it as a Dataframe
            ptsRawData = readCSV(file)
            
            # filter dataframe's rows that not contain a control point
            ptsRawData = ptsRawData.filter(like='RC', axis=0)
            
            # only selecting points that are expected
            ptsRawData = ptsRawData.loc[ptsRawData.index.intersection(expectedPtNames[mode])]
            
            # filter cases with less than 5 control points
            if ptsRawData['x'].size < 5:
                print("File {filename} failed because there was less than 5 points.".format(filename = file.split('\\')[-1].split('.')[0]))
                continue
            # fill up data structure with missing points (applyed when ...P2 points are conside red)
            # if ptsRawData['x'].size < 8:
            #     for ptName in expectedPtNames:
            #         if not ptName in ptsRawData.index:
            #             ptsRawData = ptsRawData.append(pd.DataFrame({'x': None, 'y': None, 'z':None}, index=[ptName]))           
            
            ptsRawData.sort_index(inplace=True)
            
            # filter cases where there is a point with a strange name
            # findWrongName = False
            # for ptLabel in ptsRawData.index:
            #     if ptLabel not in expectedPtNames:
            #         findWrongName = True
            #     else:
            #         if (ptLabel not in expectedPtNames):
                        
            # if findWrongName:
            #     print(filename.split('.')[0])
            #     continue
            # appending dataframes and datetime info to later construction of the main dataframe
            rawData.append(ptsRawData)
            dateRef.append(file.split('\\')[-1].split('.')[0])
    
    # concatenating dfs with raw data
    rawData = pd.concat(rawData, ignore_index=True)
    # constructing multi index structure
    combinations = [dateRef, expectedPtNames[mode]]
    index = pd.MultiIndex.from_product(combinations, names=["Run", "Point"])
    # setting multi index to df
    rawData.index = index
    # sorting by date
    # rawData.sort_index(inplace=True)

    return rawData

def calc_R(c, x, y):
    # calculate the distance of each 2D points from the center (xc, yc)
    return np.sqrt((x-c[0])**2 + (y-c[1])**2)

def circle_lstsq(c, xs, ys):
    # calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)
    Ri = calc_R(c, xs, ys)
    return Ri - Ri.mean()

def plane(x,y,A,B,D):
      return (A*x + B*y + D)
  
    
def projectPointOntoPlane(plane, pts):
    # instantiating plane parameters
    # C = -1 by default
    A,B,C,D = plane[0], plane[1], -1, plane[2] 
    
    projPts = []
    for pt in pts:
        x0,y0,z0 = pt[0], pt[1], pt[2]
        # find t that makes satisfies both plane and line equations
        t = (-A*x0 - B*y0 - C*z0 - D) / (A**2 + B**2 + C**2)
        # extract coordinates from projected point
        x = x0 + A*t
        y = y0 + B*t
        z = z0 + C*t
        # append points into list
        projPts.append([x,y,z])

    return np.row_stack(projPts)

def transform(transfMatrix, ptList):
    # initializating list that will contain transformed points
    transfPtList = []
    # iterating over original points
    for pt in ptList:
        point = np.array([[pt[0]], [pt[1]], [pt[2]], [1]])
        # applying transformation with a homogeneos matrix
        transformedPoint = transfMatrix @ point
        x = transformedPoint[0, 0]
        y = transformedPoint[1, 0]
        z = transformedPoint[2, 0]
        # appending transformed point to list
        newPoint = np.array([x,y,z])
        transfPtList.append(newPoint)

    return np.row_stack(transfPtList)

def createTransfMatrix(Tx, Ty, Tz, Rx, Ry, Rz):
    # initializating homogeneos matrix
    transfMatrix = np.zeros(shape=(4, 4))
    # defining the rotation sub-matrix
    rot_z = np.array([[np.cos(Rz), -np.sin(Rz), 0],[np.sin(Rz), np.cos(Rz), 0], [0, 0, 1]])
    rot_y = np.array([[np.cos(Ry), 0, np.sin(Ry)],[0, 1, 0], [-np.sin(Ry), 0, np.cos(Ry)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(Rx), -np.sin(Rx)], [0, np.sin(Rx), np.cos(Rx)]])
    rotationMatrix = rot_z @ rot_y @ rot_x
    # storing rotation and translation parts on the homogeneos matrix 
    transfMatrix[:3, :3] = rotationMatrix
    transfMatrix[3, 3] = 1
    transfMatrix[:3, 3] = [Tx, Ty, Tz]

    return transfMatrix


def calculateEuclidianDistance(params, *args):
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

    return np.sqrt(np.sum(diff))

def pointsBestfit(pts_corresp, pts_nom):
    # initializating array with parameters that will be evaluated (R's and T's)
    params = np.zeros(6)
    # applying minimize function to find the best suited transformation parameters
    deviation = minimize(fun=calculateEuclidianDistance, x0=params, args=(pts_corresp, pts_nom),
                         method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

    return deviation


def findPlaneRotations(normal):
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
    deviations = pointsBestfit(pts_corresponding, pts_nominal)
    rotations = deviations[3:]
    return rotations
    
def fit2DCircle(pts):
    # first estimation for the fitting
    center_init = np.nanmean(pts, axis=0)
    center_estimate = center_init[0], center_init[1]
    # isolating x and y terms
    pts = np.row_stack(pts)
    xs = pts[:, 0]
    ys = pts[:, 1]
    # filtering nan values
    xs = list(filter(lambda i: str(i) != 'nan', xs))
    ys = list(filter(lambda i: str(i) != 'nan', ys))
    # applying least square operation to fit the circle
    center, ier = leastsq(circle_lstsq, center_estimate, args=(xs, ys))
    # calculating circle's radius from the estimated center and the points
    Ri = calc_R(center, xs, ys)
    R = Ri.mean()
    return [center[0], center[1], R]

def extractFittingResults(points):
    # isolating x, y and z terms
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
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
    # projecting points onto plane 
    projPts = projectPointOntoPlane(plane_normal, points)
    # defining a transformation between the current frame and the one defined by the plane's normal and a point in the plane
    rot = findPlaneRotations(plane_normal)
    transl = projPts[0]
    # changuing point's frame/base
    transf_matrix = createTransfMatrix(transl[0], transl[1], transl[2], rot[0], rot[1], rot[2])
    projPts = transform(inv(transf_matrix), projPts)
    # fitting 2D circle
    center_x, center_y, radius = fit2DCircle(projPts)
    return [center_x, center_y, radius, projPts]

    
def processRawData(rawData, mode):
    # iniatilizating the raw structure to contain distances from the center
    distances = []
    # defining the header of the final data structure
    # header = ['Run', 'RC1_ext_P1 (mm)', 'RC2_ext_P1 (mm)', 'RC3_ext_P1 (mm)', 'RC3_ext_P2 (mm)', 'RC4_ext_P1 (mm)', 'RC4_ext_P2 (mm)', 'RC5_ext_P1 (mm)', 'RC5_ext_P2 (mm)','Mean (mm)']
    header = {'external': ['Run', 'RC1_ext_P1 (mm)', 'RC2_ext_P1 (mm)', 'RC3_ext_P1 (mm)', 'RC4_ext_P1 (mm)', 'RC5_ext_P1 (mm)','Mean (mm)'],
              'internal': ['Run', 'RC1 (mm)', 'RC2 (mm)', 'RC3 (mm)', 'RC4 (mm)', 'RC5 (mm)', 'Mean (mm)'],
              'magnets': ['Run', 'RC1_ima (mm)', 'RC2_ima (mm)', 'RC3_ima (mm)', 'RC4_ima (mm)', 'RC5_ima (mm)', 'Mean (mm)']}
    # referencing datetimes
    dateRefs = rawData.index.get_level_values(0)
    # must slice the array once each date appears 5 times
    dateRefs = dateRefs[::len(header[mode])-2]
    # iterate over datetimes and processing points of that date
    for date in dateRefs:
        # select points from that particular date
        ptList = rawData.loc[date,:].to_numpy()
        # compute circle paramaters
        results = extractFittingResults(ptList)
        center = results[:2]
        pts_proj_newBase = results[3]
        # calculate the 2D distance from each point to the center
        dateDistances = [date]
        for pt in pts_proj_newBase:
            vector = [pt[:2], center]
            # print(vector, results[2],'\n\n')
            dist = np.sqrt(np.sum(np.diff(vector, axis=0)**2))
            dateDistances.append(dist)
        mean = np.nanmean(dateDistances[1:])
        dateDistances.append(mean)
        # once the array is constructed, create an DF and append to the raw structure
        dateDistanceDF = pd.DataFrame([dateDistances], columns=header[mode])
        distances.append(dateDistanceDF)
    # concat all DFs into one
    distances = pd.concat(distances)
    distances.reset_index(drop=True, inplace=True)
    distances.set_index(header[mode][0], inplace=True)

    return distances
        

if __name__ == "__main__":
    # initiating user interactive session
    print('------------------------------\n Sirius Radius Data Generator\n------------------------------\n')
    print('Select which type of analysis it will be:')
    print('\t1. Internal')
    print('\t2. External')
    print('\t3. Magnets')



    valid_options = ['1','2','3']
    while True:
        mode_id = input('Enter option (1,2 or 3): ')
        if (mode_id not in valid_options):
            print('Option not valid, please try again.\n')
            continue

        if (mode_id == '1'):
            mode = 'internal' # or "internal"
            end_path = 'pontos_internos'
        elif (mode_id == '2'):
            mode = 'external'
            end_path = 'pontos_externos'
        elif (mode_id == '3'):
            mode = 'magnets'
            end_path = 'pontos_imas'
        break
            
    # base_path = 'R:\\LNLS\\Grupos\\GAMS\\2_Projetos\\22_Monitoramento\\Blindagem\\Raio_Parede\\dados_e_resultados\\historico\\pilar_central\\' + end_path
    base_path = os.path.join(os.path.dirname(os.getcwd()),'dados_e_resultados', 'historico', 'pilar_central', end_path)
    
    print('Path to look for the files: ' + base_path + '\\txt\n')
    print('Starting the data processing...\n')

    # processing raw data
    rawData = readAllTxtInPath(base_path+'\\txt', mode)
    distances = processRawData(rawData, mode)
    # saving into excel
    distances.to_excel(base_path+'\\radius_'+mode+'.xlsx')

    print('\nProcessing done! Excel file saved on the same path mentioned before.')


