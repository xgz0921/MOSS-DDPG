### Script for some useful functions
### @author : Guozheng Xu
### @date   : 2024-07-06
############################################################################

import os
import numpy as np

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
def write_txt(file_path,content):
    try:
        file = open(file_path, 'a')
        file.write(content)
        file.close()
    except FileNotFoundError as e:
        print(f'An error occurred: {e}')
        
def curve_fitting_SAO(metrics,bias):
    '''
    Parabola Fitting method by 2N+1 observations
    '''
    org = metrics[0]
    c_cors = []
    for i in range(metrics.shape[0]//2):
        m_p = metrics[i*2+2]
        m_n = metrics[i*2+1]
        c_cor = -bias*(m_p-m_n)/(2*m_p-4*org+2*m_n)
        c_cors.append(c_cor)
    return np.array(c_cors)