import os
from typing import Tuple
import numpy as np
import pandas as pd

from geometric import create_new_frame_and_transform_points, fit_2D_circle, fit_plane_and_extract_normal, project_point_onto_plane
from io_operations import read_point_files_and_compile_data
import config


def compute_circle_fitting(points):
    # fitting a plane
    plane_normal = fit_plane_and_extract_normal(points[:,0], points[:,1], points[:,2])
    # projecting points onto plane 
    projPts = project_point_onto_plane(plane_normal, points)
    # create new frame with directions determined by the plane normal and with origin somewhere in the plane
    projPts_transformed = create_new_frame_and_transform_points(plane_normal, projPts[0], projPts)
    # fitting 2D circle
    center, _ = fit_2D_circle(projPts_transformed)
    return center, projPts_transformed

def calculate_radius(raw_data: pd.DataFrame, mode: str) -> pd.DataFrame:
    # iniatilizating the raw structure to contain radius values
    radius = []

    # defining the header of the final data structure
    header = {k: ['Run'] + [f'{i} (mm)' for i in config.EXPECTED_POINT_NAMES[k]] + ['Mean (mm)'] for k in config.EXPECTED_POINT_NAMES}

    # referencing datetimes
    date_refs = raw_data.index.get_level_values(0)

    # must slice the array once each date appears 5 times
    date_refs = date_refs[::len(header[mode])-2]
    
    # iterate over datetimes and processing points of that date
    for date in date_refs:
        # select points from that particular date
        points = raw_data.loc[date,:].to_numpy()

        # compute fitted circle paramaters
        center, projPts_transformed = compute_circle_fitting(points)

        # calculate the 2D distance from each point to the center and its average
        date_radius = [date]
        for pt in projPts_transformed:
            vector = [pt[:2], center]
            dist = np.sqrt(np.sum(np.diff(vector, axis=0)**2))
            date_radius.append(dist)
        mean = np.nanmean(date_radius[1:])
        date_radius.append(mean)

        # once the array is constructed, create an DF and append to the raw structure
        date_radius_df = pd.DataFrame([date_radius], columns=header[mode])
        radius.append(date_radius_df)

    # concat all DFs into one
    radius = pd.concat(radius)
    radius.reset_index(drop=True, inplace=True)
    radius.set_index(header[mode][0], inplace=True)

    return radius
        
def show_user_options() -> Tuple[str, str]:
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
            mode = 'internal'
            end_path = 'pontos_internos'
        elif (mode_id == '2'):
            mode = 'external'
            end_path = 'pontos_externos'
        elif (mode_id == '3'):
            mode = 'magnets'
            end_path = 'pontos_imas'
        break

    return (mode, end_path)

if __name__ == "__main__":
    # initiating user interactive session
    mode, end_path = show_user_options()
    
    # defining path to read and write files
    base_path = os.path.join(config.BASE_PATH, end_path)

    print(f'Path to look for the files: {base_path}\\txt\n')
    print('Starting the data processing...\n')

    # processing raw data
    raw_data = read_point_files_and_compile_data(os.path.join(base_path, 'txt'), mode)
    radius = calculate_radius(raw_data, mode)

    # saving into excel
    radius.to_excel(base_path+'\\radius_'+mode+'.xlsx')

    print('\nProcessing done! Excel file saved on the same path mentioned before.')


