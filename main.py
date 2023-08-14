import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import argparse
import multiprocessing as mp
from algorithm import fedavg, fedprox, hierfl, fedasync
from tools import data_distributer
import numpy as np
import logging
import os
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="hierfl")
    parser.add_argument('--datapath', type=str, default='../../data')
    parser.add_argument('--savepath', type=str, default='./task_record/')
    parser.add_argument('--dataset', type=str, default='cifar10') # see tools.utils.py
    parser.add_argument('--modelname', type=str, default='cifarcnn') # see tools.utils.py
    parser.add_argument('--clients', type=int, default=21)
    parser.add_argument('--seed', type=int, default=2222)
    # parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--sample_ratio', type=float, default=0.5)
    parser.add_argument('--aggregate_methods', type=str, default="weight") # or uniform
    parser.add_argument('--batchsizes', type=float, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--partition', type=str, default="niid") # iid,niid,shard
    parser.add_argument('--dir', type=float, default=0.5)
    parser.add_argument('--shards', type=float, default=2)
    parser.add_argument('--gpu_nums', type=int, default=4)
    args = parser.parse_args()
    return args

def get_logger(args):

    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    if args.dir!=0:
        savelog='%s_%s_%s_%s_Clt%d_SR%.2f_R%d_EP%d_Dir%.1f_lr%.2f'%\
                (args.algorithm,args.dataset,
                args.modelname, args.partition, args.clients, args.sample_ratio,
                 args.rounds, args.epochs, args.dir, args.lr)
    else:
        savelog = '%s_%s_%s_%s_Clt%d_SR%.2f_R%d_EP%d_Shard%d_lr%.2f' % \
                  (args.algorithm, args.dataset,
                  args.modelname, args.partition, args.clients,  args.sample_ratio,
                  args.rounds, args.epochs, args.shards, args.lr)

    if os.path.exists(args.savepath+savelog):
        print(f"Path '{args.savepath+savelog}' exists.")
        os.remove(args.savepath+savelog+"/%s.log"%savelog)
    else:
        print(f"Mklog '{args.savepath+savelog}'.")
        os.makedirs(args.savepath+savelog)

    fwrite = logging.FileHandler(args.savepath+savelog+"/%s.log"%savelog)
    fwrite.setLevel(logging.INFO)

    printer = logging.StreamHandler()
    printer.setLevel(logging.DEBUG)

    # 给logger添加handler
    logger.addHandler(fwrite)
    logger.addHandler(printer)

    return logger

if __name__=="__main__":
    args = get_args()
    print(args)
    args.algorithms = {
        "fedavg": [fedavg,{}],
        "fedprox": [fedprox, {"mu": 0.01}],
        "hierfl":[hierfl, {"tiers":5,"flag":'poly',"alpha":0.2}],
        "fedasync": [fedasync, {"flag": 'poly', "alpha": 0.3}]
    }
    processes = []
    # start training
    args.distributer = data_distributer(args.datapath,
                                   args.dataset,
                                   args.batchsizes,
                                   args.clients-1,
                                   args.partition,
                                        args.dir,
                                        args.shards)
    data_map=args.distributer["data_map"]
    args.data_ratio=np.sum(data_map,axis=1)/np.sum(data_map)
    args.logger=get_logger(args)
    args.logger.info("=============================================================")
    args.logger.info("Client data ratio: %s"%(str(args.data_ratio)))
    args.logger.info("=============================================================")
    for c in range(args.clients):
        if c == args.clients-1:
            # -------------------------start server--------------------------------
            S = args.algorithms[args.algorithm][0].Server(args)
            p = mp.Process(target=S.run, args=())
        else:
            # -------------------------start client--------------------------------
            C = args.algorithms[args.algorithm][0].Client(c, args)
            p = mp.Process(target=C.run, args=())
        p.start()
        processes.append(p)

    # stop training
    processes[-1].join()
    if not processes[-1].is_alive():
        for p in processes[:-1]:
            p.terminate()

