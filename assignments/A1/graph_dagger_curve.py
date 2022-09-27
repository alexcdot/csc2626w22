import matplotlib.pyplot as plt
import argparse
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggr_errors_file", type=str, help="location of saved errors json", required=True)
    parser.add_argument("--out_dir", type=str, help="where to save results", default="")
    args = parser.parse_args()

    aggr_errors = json.load(open(args.aggr_errors_file))

    for k, values in aggr_errors:
        plt.plot(range(len(values)), values, marker="o", linewidth=2)

        plt.savefig(os.path.join(args.out_dir, k + ".pdf"))