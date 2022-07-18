import os
import re
import pandas as pd
import shutil


results_folder = "../results/"

final_data = []
for file in os.listdir(results_folder):
    if ".csv" in file and "losses" not in file:
        print("Curr file: ", file)
        df = pd.read_csv(os.path.join(results_folder, file))
        mean_dl = df["similarity"].mean()
        log, fold, algorithm = re.match(r'(.*)_EXPERIMENTACION_TFG_.*_fold_(\d+)_postprocessing_(.*).csv', file).groups()
        final_data.append(
            {
                "log" : log,
                "fold" : fold,
                "mean_dl" : mean_dl,
                "algorithm" : algorithm
            }
        )

df = pd.DataFrame(final_data)
print("Dataframe: ")
print(df)
print("Grouped results")
print(df.groupby(["log", "algorithm"]).mean())
