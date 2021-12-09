import config

import os
import pandas as pd

def readCSV(fileDir) -> pd.DataFrame:

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

def read_point_files_and_compile_data(path: str, mode: str) -> pd.DataFrame:
    rawData, dateRef = [], []
    expectedPtNames = config.EXPECTED_POINT_NAMES

    # listing all files in path
    files = os.listdir(path)
    files_full_path = [os.path.join(path, file) for file in files]

    # ordering list according to modification time
    sorted_files = sorted(files_full_path, key=lambda f: os.stat(f).st_ctime)

    # iterating over all files in the path
    for file in sorted_files:
        
        if file.endswith('.txt'):
            
            ptsRawData = readCSV(file)
            
            # filter dataframe's rows that not contain a control point
            ptsRawData = ptsRawData.filter(like='RC', axis=0)
            
            # only selecting points that are expected
            ptsRawData = ptsRawData.loc[ptsRawData.index.intersection(expectedPtNames[mode])]
            
            # filter cases with less than 5 control points
            if ptsRawData['x'].size < 5:
                print("File {filename} failed because there was less than 5 points.".format(filename = file.split('\\')[-1].split('.')[0]))
                continue          
            
            # sorting dataframe by date
            ptsRawData.sort_index(inplace=True)
            
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

    return rawData