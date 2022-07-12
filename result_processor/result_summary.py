import os
import re
import pandas as pd
import shutil


results_folder = "./results_no_positional/"

final_data = []
for file in os.listdir(results_folder):
    if ".csv" in file and "losses" not in file:
        df = pd.read_csv(os.path.join(results_folder, file))
        mean_dl = df["similarity"].mean()
        log, fold = re.match(r'(.*)_experimentation_tfg_.*_fold_(\d+)', file).groups()
        final_data.append(
            {
                "log"  : log,
                "fold" : fold,
                "mean_dl" : mean_dl
            }
        )

df = pd.DataFrame(final_data)
print("Grouped results")
print(df.groupby(["log"]).mean())