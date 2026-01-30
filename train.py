# -*- coding: utf-8 -*-

import argparse
import torch
from par import pars
from topology import Topology
from matrix import make_matrix, make_strongly_connected
from attack import attacks
import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

def train(epochs, logdir, dataset, batch_size, noniid_rate, adj_matrix, attacks, par, args):
    topo = Topology(epochs, logdir, adj_matrix, attacks, par=par)
    topo.build_topo(dataset, batch_size, noniid_rate, args)
    for worker in topo.workers:
        worker.start()
    for worker in topo.workers:
        worker.join()


if __name__ == "__main__":
    # spawn method for cuda
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help="MNIST, FashionMNIST: 50, FEMNIST: 100, CIFAR10: 150"
    )
    parser.add_argument("--dataset", type=str, default="FashionMNIST")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--meta_lr", type=float, default=1e-3)
    parser.add_argument("--nodes_n", type=int, default=10)
    parser.add_argument("--byzantine_ratio", type=float, default=0.3)
    parser.add_argument("--connection_ratio", type=float, default=0.5) 
    parser.add_argument("--attack", type=str, default="gaussian")
    parser.add_argument("--par", type=str, default="felm_da")
    parser.add_argument("--logdir", type=str, default="test")
    parser.add_argument("--noniid_rate", type=float, default=0.5)
    args = parser.parse_args()

    adj_matrix, attacks = make_matrix(nodes_n=args.nodes_n, 
                                  connect_probs=args.connection_ratio, 
                                  byzantine_probs=args.byzantine_ratio, 
                                  attack=args.attack)
    adj_matrix, edges_added = make_strongly_connected(adj_matrix)

    print(adj_matrix)
    print(attacks)
    workers_n = args.nodes_n
    par_args = {
        "lr": 1e-4,
        "gamma": 0.98,
        "batch_size": 512,
        "restore_path": "models/"
    }
    train(
        args.epochs,
        args.logdir,
        args.dataset,
        args.batch_size,
        args.noniid_rate,
        adj_matrix=adj_matrix,
        attacks=attacks,
        par=pars[args.par],
        args=par_args,
    )
