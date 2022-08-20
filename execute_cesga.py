import argparse
import os

parser = argparse.ArgumentParser(description="Prepare to execute in CESGA")
parser.add_argument("--train", action="store_true")
parser.add_argument("--postprocessing", type=str, required=True, choices=["beam", "beam_length_normalized", "beam_length_normalized_coverage", "beam_monteagudo", "argmax", "random"])
parser.add_argument("--disable_attention", required=False, action="store_true")
parser.add_argument("--optimize_beam_width", required=False, action="store_true")
parser.add_argument("--store_prefixes_in_results", required=False, action="store_true")
args = parser.parse_args()

for file in os.listdir("./data/data_complete"):
    if "csv" in file:
        base = file.replace(".csv", "")
        for i in range(5):
            fold_num = i
            command = f"sbatch ./execute_job.sh --dataset {base} --execution_id EXPERIMENTACION_TFG --fold_num {i}"
            if args.train:
                command += " --train"
            if args.postprocessing:
                command += f" --postprocessing {args.postprocessing}"
            if args.disable_attention:
                command += " --disable_attention"
            if args.optimize_beam_width:
                command += " --optimize_beam_width"
            if args.store_prefixes_in_results:
                command += " --store_prefixes_in_results"
            print("==========")
            print(command)
            os.system(command)
            print("==========")
