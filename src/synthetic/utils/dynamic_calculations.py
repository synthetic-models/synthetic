# imports 
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def last_time_point_method(time_course_data, selected_species=None):
    if selected_species is None:
        selected_species = time_course_data.columns
    else:
        selected_species = selected_species
    selected_time_course_data = time_course_data[selected_species]
    last_time_points = selected_time_course_data.map(lambda x: x[-1])
    return last_time_points

def get_dynamic_features(col_data: pd.Series, 
                         normalise: bool = True,
                         abs_change_tolerance: float = 0.01) -> list:
    
    # dynamic features
    auc = np.trapz(col_data)
    max_val = np.max(col_data)
    max_time = np.argmax(col_data)
    min_val = np.min(col_data)
    min_time = np.argmin(col_data)

    median_val = np.median(col_data)

    # calculation of total fold change (tfc)
    start_val = col_data.iloc[0]
    end_val = col_data.iloc[-1]

    tfc = 0 
    if start_val == 0:
        tfc = 1000
    else:
        if end_val - start_val >= 0:
            tfc = (end_val - start_val) / start_val
        elif end_val - start_val < 0:
            if end_val == 0:
                tfc = -1000
            else:
                tfc = -((start_val - end_val) / end_val)

    # calculation of time to stability (tsv)
    tsv = len(col_data)
    while tsv > 1:
        if abs(col_data.iloc[tsv-1] - col_data.iloc[tsv-2]) < abs_change_tolerance:
            tsv -= 1
        else:
            tsv_value = col_data.iloc[tsv-1]
            break
    if tsv == 1:
        tsv_value = col_data.iloc[0]

    max_sim_time = len(col_data)
    n_auc = auc / max_sim_time
    n_max_time = max_time / max_sim_time
    n_min_time = min_time / max_sim_time
    n_tsv = tsv / max_sim_time
    
    if not normalise:
        # reset the values to the original values
        n_auc = auc
        n_max_time = max_time
        n_min_time = min_time
        n_tsv = tsv 

    dynamic_features = [n_auc, median_val, tfc, n_max_time,
                        max_val, n_min_time, min_val, n_tsv, tsv_value, start_val]

    return dynamic_features


def dynamic_features_method(
    time_course_data, selected_features=None, n_cores=1, verbose=0
):
    if selected_features is None:
        selected_features = time_course_data.columns
    else:
        selected_features = selected_features

    # use parallel processing to speed up the calculation with tqdm
    def process_data(row_data):
        row_dynamic_features = []
        for feature in selected_features:
            col_data = row_data[feature]
            # convert to pd Series for easier manipulation
            col_data = pd.Series(col_data)
            dyn_feats = get_dynamic_features(col_data)
            row_dynamic_features.extend(dyn_feats)
        return row_dynamic_features

    if n_cores > 1 or n_cores == -1:
        all_dynamic_features = Parallel(n_jobs=n_cores)(
            delayed(process_data)(time_course_data.iloc[i])
            for i in tqdm(
                range(time_course_data.shape[0]),
                desc="Calculating dynamic features",
                disable=verbose == 0,
            )
        )
    else:
        all_dynamic_features = []
        # iterate each row in the time course data
        for i in tqdm(
            range(time_course_data.shape[0]),
            desc="Calculating dynamic features",
            disable=verbose == 0,
        ):
            row_dynamic_features = []
            row_data = time_course_data.iloc[i]
            for feature in selected_features:
                col_data = row_data[feature]
                # convert to pd Series for easier manipulation
                col_data = pd.Series(col_data)
                dyn_feats = get_dynamic_features(col_data)
                row_dynamic_features.extend(dyn_feats)
            all_dynamic_features.append(row_dynamic_features)

    dynamic_feature_label = [
        "auc",
        "median",
        "tfc",
        "tmax",
        "max",
        "tmin",
        "min",
        "ttsv",
        "tsv",
        "init",
    ]
    new_df = pd.DataFrame(
        all_dynamic_features,
        columns=[
            s + "_" + dynamic_feature
            for s in selected_features
            for dynamic_feature in dynamic_feature_label
        ],
        index=time_course_data.index,
    )

    return new_df
