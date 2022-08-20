import os
import re
import pandas as pd
import shutil


folder_output_path = "./data/paper_results/results/"
input_result_path = "./results_attention/"

final_data = []
for file in os.listdir(input_result_path):
    if ".csv" in file and "losses" not in file:
        df = pd.read_csv(os.path.join(input_result_path, file))
        print(file)
        mean_dl = df["similarity"].mean()
        log, fold = re.match(r'(.*)_test.*_fold_(\d+)_.*', file).groups()
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

other_results = pd.read_csv("./data/paper_results/accuracy_raw_results_others.csv")
other_results = other_results.drop(columns=["Unnamed: 0"])
other_results = other_results.rename(columns={"accuracy": "mean_dl"})
ours_df = df
ours_df["log"] = ours_df["log"].str.lower()
ours_df["approach"] = "ABASP"
merged_df = pd.concat([ours_df, other_results])
merged_df = merged_df.groupby(["approach"])
print("==========================================================")
print("MEAN")
print("==========================================================")
print(merged_df.mean())
print("==========================================================")
print("STD")
print("==========================================================")
print(merged_df.std())

if os.path.exists(folder_output_path):
    shutil.rmtree(folder_output_path)

os.makedirs(folder_output_path, exist_ok=True)
for data in final_data:
    specific_folder = os.path.join(folder_output_path, "train_fold" + data["fold"] + "_variation0_" + data["log"]+ ".xes.gz")
    os.makedirs(specific_folder, exist_ok=True)
    with open(os.path.join(specific_folder, "ggrnn_results.txt"), "w") as result_file:
        result_file.write("Accuracy: " + str(data["mean_dl"]))

