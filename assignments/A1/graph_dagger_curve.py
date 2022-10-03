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

    plt.plot(range(1, len(aggr_errors["aggr_error_dists"]) + 1), aggr_errors["aggr_error_dists"], marker="o", linewidth=2)
    plt.xlabel("Number of dagger iterations")
    plt.ylabel("Cumulative distance error")
    plt.savefig(os.path.join(args.out_dir, "aggr_error_dists.pdf"))

    plt.cla()

    plt.plot(range(1, len(aggr_errors["aggr_error_headings"]) + 1), aggr_errors["aggr_error_headings"], marker="o", linewidth=2)
    plt.xlabel("Number of dagger iterations")
    plt.ylabel("Cumulative heading error")
    plt.savefig(os.path.join(args.out_dir, "aggr_error_headings.pdf"))

    plt.cla()

    plt.plot(range(1, len(aggr_errors["aggr_num_steps"]) + 1), aggr_errors["aggr_num_steps"], marker="o", linewidth=2)
    plt.xlabel("Number of dagger iterations")
    plt.ylabel("Number of steps successfully taken in environment")
    
    plt.savefig(os.path.join(args.out_dir, "aggr_num_steps.pdf"))

    plt.cla()
    dist_per_step = [(dist + 0.0) / steps for steps, dist in zip(aggr_errors["aggr_num_steps"], aggr_errors["aggr_error_dists"])]

    plt.plot(range(1, len(dist_per_step) + 1), dist_per_step, marker="o", linewidth=2)
    plt.xlabel("Number of dagger iterations")
    plt.ylabel("Amount of error per step")
    
    plt.savefig(os.path.join(args.out_dir, "dist_per_step.pdf"))

    