#SVD point identification, return the first point that the 
#real-time detection value exceeds 0.3(threshold), it's subject to adjustment.
#This script should be in the same directory as that of the SVD results datasets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def identify_detection_points(threshold):
    proc_dir = Path('.')
    files = sorted([p for p in proc_dir.glob("*_SVD_detects.csv") ])
    df_result = pd.DataFrame({
      'dataset_name': [],
      'detection_point': []
    })
    for f in files:
        df = pd.read_csv(f)
        dataset_name = str(f).split('_')[0]
        if len(np.where(df['detected_value_rt'] >= 0.3)[0]) > 0:
            detection_point = np.where(df['detected_value_rt'] >= 0.3)[0][0]
        else:
            detection_point = np.nan
        new_row = pd.DataFrame({'dataset_name': [dataset_name],
            'detection_point': [detection_point]})
        df_result = pd.concat([df_result, new_row], ignore_index = True)
    df_result.to_csv("detection_point_trigger.csv")
    print(f"Processing complete. Threshold used: {threshold}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Identify SVD detection points based on a threshold.')
    parser.add_argument('threshold', type = float, help = 'The detection threshold value (e.g., 0.3)')
    args = parser.parse_args()
    identify_detection_points(args.threshold)

#Call this function:   python detect.py 0.3