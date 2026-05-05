#SVD point identification, return the first point that the 
#real-time detection value exceeds 0.3(threshold, as a hyperparameter here.), it's subject to adjustment.
#This script should be in the same directory as that of the SVD results datasets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def identify_detection_points(threshold):
    SVD_dir = Path('SVD_result')
    T_dir = Path('T_result')
    files_SVD = sorted([p for p in proc_dir.glob("*_SVD_detects.csv") ])
    df_result = pd.DataFrame({
        'dataset_name': [],
        'detection_point': [],
        'trigger_temperature': [],
        'window_index': []
    })
    files_T = sorted([p for p in T_dir.glob("*_rt_detect_result_window100.csv")]) 
    for f in files_SVD:
        df = pd.read_csv(f)
        dataset_name = str(f).split('/')[1].split('_')[0]
        if len(np.where(df['detected_value_rt'] >= 0.3)[0]) > 0:
            detection_point = np.where(df['detected_value_rt'] >= 0.3)[0][0].item()
        else:
            detection_point = np.nan
        #Finding the exact match points:
        matches = [g.name for g in files_T if g.name.startswith(dataset_name)]
        df_T = pd.read_csv(T_dir/matches[0])
        if detection_point > 0 and detection_point < df_T.shape[0]:
            trigger_temperature = df_T.loc[detection_point - 30, "temperature"].item()
        else:
            trigger_temperature = np.nan
        #extract the window index
        if detection_point > 0:
            window_index = (detection_point - 100)//30
        else:
            window_index = np.nan
        #get the 
        new_row = pd.DataFrame({'dataset_name': [dataset_name],
            'detection_point': [detection_point],
            'trigger_temperature': [trigger_temperature],
            'window_index': [window_index]})
        df_result = pd.concat([df_result, new_row], ignore_index = True)
        df_result.to_csv("detection_point_trigger.csv")
        print(f"Processing complete. Threshold used: {threshold}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Identify SVD detection points based on a threshold.')
    parser.add_argument('threshold', type = float, help = 'The detection threshold value (e.g., 0.3)')
    args = parser.parse_args()
    identify_detection_points(threshold)

#Call this function:   python SVD_point_identification.py 0.3