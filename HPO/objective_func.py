from optuna import Trial
import numpy as np
import joblib
import os
import optuna
from torch import multiprocessing as mp
from .hpo_utils import get_hp_list
import csv


HPO_ALS = {"tpe": optuna.samplers.TPESampler,
           "cma_es": optuna.samplers.CmaEsSampler,
           "bo": optuna.integration.BoTorchSampler}


def objective(train_set, val_set, model, trial: Trial, worker, DDP: bool, args):
    # one trial
    processes = []
    if DDP:
        eval_history = mp.Manager().dict()  # Must be declared from torch multiprocessing
        for rank in range(args.world_size):
            args.local_rank = rank
            get_hp_list(args=args, rank=rank, trial=trial)
            p = mp.Process(target=worker,
                           args=(train_set, val_set, model, args, DDP, eval_history, trial)
                           )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        for p in processes:
            p.close()
    else:
        eval_history = {}
        get_hp_list(args=args, rank=args.local_rank, trial=trial)
        worker(train_set, val_set, model, args, DDP, eval_history, trial)

    '''------log lr versus val acc------'''
    filename = os.path.join(args.job_dir, 'lr_versus_eval.csv')
    fields = ['lr', 'val_acc']

    with open(filename, 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerow([args.lr, np.max(eval_history["values"])])
    csvfile.close()
    '''------log lr versus val acc------'''

    return np.max(eval_history["values"])


def do_hpo(args, train_set, val_set, model, DDP, worker):

    if not os.path.exists(args.job_dir):
        os.mkdir(args.job_dir)

    study_storage = os.path.join(args.job_dir, args.job_name + f"_{args.seed}.pkl")

    if os.path.isfile(study_storage):

        print("------------Loading your existing HPO job--------------")

        hp_finder = joblib.load(study_storage)

    else:
        print("------------Create new HPO job-------------------------")
        print(f"Hyperparameter Optimization Algorithm : {args.hpo_name}")
        print("Search space pruner is hyperband")
        print(f"The HPO jobs are spread out in {args.world_size}-GPU")

        sampler = HPO_ALS[args.hpo_name]()
        hp_finder = optuna.create_study(sampler=sampler,
                                        pruner=optuna.pruners.HyperbandPruner(),
                                        study_name=args.job_name,  # User option 을 이용해서 unique name 만들기
                                        direction='maximize')

        joblib.dump(hp_finder, study_storage)

    print("------------Start HPO-------------------------")
    # train_set, val_set, model, trial: Trial, worker, DDP: bool, args
    hp_finder.optimize(lambda trial: objective(train_set=train_set,
                                               val_set=val_set,
                                               model=model, trial=trial,
                                               worker=worker, DDP=DDP, args=args),
                       n_trials=args.num_trials)

    best_eval_value = hp_finder.best_trial.value
    best_hps = hp_finder.best_trial.params.items()  # key : lr, bs, wd
    print("------------HPO is finished--------------")
    result_log = {"best_eval_value": best_eval_value,
                  "best_hps": best_hps}
    print(result_log)
    print("------------HPO result is updated in your configuration file--------------")
