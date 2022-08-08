import os
import re
import pandas as pd
import numpy as np

attention_result_folder = "../results_attention/"
no_attention_result_folder = "../results_attention/"

full_results = []
for file in os.listdir(attention_result_folder):
    if ".csv" in file and "losses" not in file:
        df = pd.read_csv(os.path.join(attention_result_folder, file))
        mean_dl = df["similarity"].mean()
        log, fold, algorithm = re.match(r'(.*)_EXPERIMENTACION_TFG_.*_fold_(\d+)_postprocessing_(.*).csv', file).groups()
        for row in df.itertuples():
            prediction = np.fromstring(row.prediction[1:-1], dtype=int, sep=",")
            truth = np.fromstring(row.truth[1:-1], dtype=int, sep=",")
            full_results.append({
                "prediction" : prediction,
                "prediction_length" : len(prediction),
                "truth" : truth,
                "truth_length" : len(truth),
                "similarity" : row.similarity,
                "log" : log,
                "fold" : fold,
                "algorithm" : algorithm,
                "attention" : True
            })

for file in os.listdir(no_attention_result_folder):
    if ".csv" in file and "losses" not in file:
        df = pd.read_csv(os.path.join(no_attention_result_folder, file))
        mean_dl = df["similarity"].mean()
        log, fold, algorithm = re.match(r'(.*)_EXPERIMENTACION_TFG_.*_fold_(\d+)_postprocessing_(.*).csv', file).groups()
        for row in df.itertuples():
            prediction = np.fromstring(row.prediction[1:-1], dtype=int, sep=",")
            truth = np.fromstring(row.truth[1:-1], dtype=int, sep=",")
            full_results.append({
                "prediction" : prediction,
                "prediction_length" : len(prediction),
                "truth" : truth,
                "truth_length" : len(truth),
                "similarity" : row.similarity,
                "log" : log,
                "fold" : fold,
                "algorithm" : algorithm,
                "attention" : False
            })

df = pd.DataFrame(full_results)
subset = df[(df["log"] == "BPI_Challenge_2012_A") & (df["fold"] == "0")]
pd.set_option('display.max_columns', None)
print(subset)
