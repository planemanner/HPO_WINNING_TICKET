from optuna import Trial


def get_hp_list(args, rank, trial: Trial):
    # 새로운 HP 획득

    args.lr = trial.suggest_float(name="lr",
                                  low=args.lr_min,
                                  high=args.lr_max)

    # args.bs = trial.suggest_int(name="bs",
    #                             low=args.bs_min,
    #                             high=args.bs_max)
    #
    # args.wd = trial.suggest_float(name="wd",
    #                               low=args.wd_min,
    #                               high=args.wd_max)
    if rank == 0:
        print("----------------------------------------------------------------------------------------")
        print(f"| Sampled Batch size : {args.bs}, Learning rate : {args.lr}, Weight decay : {args.wd} |")
        print("----------------------------------------------------------------------------------------")