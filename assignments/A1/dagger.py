from distutils.log import error
from utils import str2bool
import train_policy
import racer
import argparse
import os
import shutil
import json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", type=int, help="", default=10)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)
    parser.add_argument("--min_turns_per_batch", type=int, help="minimum number of turning scenes per batch",
                        default=0)
    args = parser.parse_args()

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####
    dagger_train_dir = args.train_dir + f"_dagger_iters_{args.dagger_iterations}_epochs_{args.n_epochs}"
    if not os.path.exists(dagger_train_dir):
        os.mkdir(dagger_train_dir)
        shutil.copytree(args.train_dir, dagger_train_dir, dirs_exist_ok=True) 

    args.train_dir = dagger_train_dir

    aggr_error_headings = []
    aggr_error_dists = []

    for dagger_iter in range(args.dagger_iterations + 1):
        if dagger_iter == 0:
            print ('\nTRAINING LEARNER ON INITIAL DATASET')
        else:
            print ('\nRETRAINING LEARNER ON AGGREGATED DATASET, ITER', dagger_iter)
    
        args.weights_out_file = f"./weights/learner_{dagger_iter}.weights"
        steering_network = train_policy.main(args)

        if dagger_iter == args.dagger_iterations - 1:
            continue
        print ('GETTING EXPERT DEMONSTRATIONS')

        dagger_parser = racer.get_dagger_parser()
        dagger_args = dagger_parser.parse_args([
            "--out_dir", dagger_train_dir,
            "--save_expert_actions", "True",
            "--run_id", str(dagger_iter + 1),
            "--n_steering_classes", str(args.n_steering_classes)
        ])
        error_headings, error_dists = racer.run(steering_network, dagger_args)
        error_heading = sum(error_headings) / len(error_headings)
        error_dist = sum(error_dists) / len(error_dists)
        print(f"Iter: {dagger_iter} | Error heading: {error_heading} error distance: {error_dist}")
        aggr_error_headings.append(error_heading)
        aggr_error_dists.append(error_dist)

    aggr_errors = {"aggr_error_headings": aggr_error_dists, "aggr_error_dists": aggr_error_dists}
    print(aggr_errors)

    json.dump(aggr_errors, open(f"./outputs_dagger_iters_{args.dagger_iterations}_epochs_{args.n_epochs}", "w"))

            
