#compute mean loading:

from pathlib import Path
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
'''
The input would be including the 
X: n * 3 dimensional array, include the temperature, voltage as well as the force
doing the running window-size with different steps.
for each of the, return the proportion of the loading along each of the dimension as the value/proportion.
'''
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def compute_mean_loading_prop(X, window, step):
    n = X.shape[0]
    n_window = len(np.arange(window, X.shape[0], step))
    prop_result = np.random.random((n_window, 3))
    window_indices = list(np.arange(window, X.shape[0], step))
    for i in range(len(window_indices)):
        window_df = X[(window_indices[i]-window):window_indices[i],:]
        mu = window_df.mean(axis = 0)
        sd = window_df.std(axis = 0)
        sd[sd < 1e-12] = 1.0
        z = (window_df - mu)/sd
        pca = PCA()
        pca.fit(z)
        pc1_loadings = pca.components_[0]
        pc1_proportions = pc1_loadings ** 2
        prop_result[i,:] = pc1_proportions
        #weight
    result_df = pd.DataFrame(
    {
    'Window_Index': np.arange(1, len(prop_result) + 1),
    'Loading_Temperature_Prop': pd.Series(prop_result[:,0], dtype = float),
    'Loading_force_Prop': pd.Series(prop_result[:,1], dtype = float),
    'Loading_Voltage_Prop': pd.Series(prop_result[:,2], dtype = float),
    })
    return result_df


#calculating the mean loading trends:
src_dir = Path("processed_data")
files = sorted(
[p for p in src_dir.glob("*.csv")
if ("xlsx_reform_severity_level" in p.name or ".xlsx" in p.name)])
#The files are the list of them illustrating the severity levels

#extract all of the datasets and visualize them,
#the window size and the step-size are the points.
WINDOW = 200
STEP = 15
if not os.path.exists(f"mean_loading_prop_analysis/mean_{WINDOW}_{STEP}"):
    os.mkdir(f"mean_loading_prop_analysis/{WINDOW}_{STEP}")
for i in range(len(files)):
    f = files[i]
    df = pd.read_csv(f)
    df_name = str(str(f).split("/")[1].split(".")[0])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis = 1)
    X = np.array(df[['temperature', 'load', 'voltage']])
    loading_prop_df = compute_mean_loading_prop(X,  window = WINDOW, step = STEP)
    loading_prop_df.to_csv(f"mean_loading_prop_analysis/{WINDOW}_{STEP}/{df_name}" + f"loading_prop_{WINDOW}_{STEP}.csv")
    #Draw the plot here:
    df_long = loading_prop_df.melt(
        id_vars = ["Window_Index"],
        value_vars = [
        'Loading_Temperature_Prop',
        'Loading_force_Prop',
        'Loading_Voltage_Prop'],
        var_name = 'Direction',
        value_name = 'Loading_Proportion')
    name_map = {
    'Loading_Temperature_Prop': 'Temperature',
    'Loading_force_Prop': 'Force',
    'Loading_Voltage_Prop': 'Voltage'
    }
    df_long['Direction'] = df_long['Direction'].map(name_map)
    # 2. Setup Theme
    sns.set_theme(style='whitegrid', font_scale=1.1)
    plt.figure(figsize=(11, 6))
    # 3. Create Plot
    ax = sns.lineplot(
        data=df_long,
        x="Window_Index",
        y="Loading_Proportion",
        hue='Direction',
        linewidth=1.25,
        palette='viridis' # Professional, color-blind friendly palette
    )
    # 4. Polish the Legend
    # 'bbox_to_anchor' moves it outside the plot area
    sns.move_legend(
        ax, "upper left", 
        bbox_to_anchor=(1, 1), 
        title='Prop. Type', 
        frameon=False
    )
    # 5. Labels and Titles
    ax.set_title(f"Loading Proportion Trend: {df_name}\n(Window: {WINDOW}, Step: {STEP})", 
                 fontsize=14, pad=15, fontweight='bold')
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Loading Proportion', fontsize=12)
    # 6. Final Touches
    sns.despine() # Removes the top and right spines for a cleaner look
    plt.tight_layout()
    # Save
    save_path = Path(f"mean_loading_prop_analysis/{WINDOW}_{STEP}")
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"LoadingProportion_{df_name}.png", dpi=300, bbox_inches='tight')











