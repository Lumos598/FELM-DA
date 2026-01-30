import logging
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.multiprocessing import Process
from utils import (
    TO_CUDA,
    check_dir,
    collect_grads,
    get_meta_model_flat_params,
    meta_test,
    set_grads,
    set_meta_model_flat_params,
    get_meta_model_params_dict,
    get_meta_model_classes_params_dict,
)
import numpy as np


label_len = {
    "MNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "FashionMNIST": 10,
    "FEMNIST": 62
}


class Worker(Process):
    MASTER_ADDR = "127.0.0.1"
    MASTER_PORT = "20056"

    def __init__(
            self,
            rank,
            size,
            attack,
            test_ranks,
            meta_lr,
            train_loader,
            test_loader,
            model,
            dataset="MNIST",
            criterion=F.cross_entropy,
            epochs=100,
            weight_decay=5e-4,
            logdir="test",
    ) -> None:
        super().__init__()
        self.logdir = os.path.join("result/", logdir)
        self.rank = rank

        if self.rank < 35:
            self.device_id = 0
        else:
            self.device_id = 1

        self.size = size
        self.attack = attack
        self.criterion = criterion
        self._train_loader = train_loader
        self._test_loader = test_loader
        self.dataset = dataset
        self.meta_model=model
        self.optimizer = optim.Adam(
            self.meta_model.parameters(), lr=meta_lr, weight_decay=weight_decay
        )
        self.test_ranks = test_ranks
        # others -> self
        self.src = []
        # self -> others
        self.dst = []
        # training params
        self.epochs = epochs
        self.par = None
        self.num_byzantine = 0

    def construct_src_and_dst(self):
        if self.attack is None:
            # non-byzantine worker extend the dst
            for rank in range(self.size):
                if (rank not in self.dst) and (rank not in self.test_ranks):
                    self.dst.append(rank)
        else:
            # byzantine worker clean the original src and extend the src
            self.src = self.test_ranks
            for rank in range(self.size):
                if (rank in self.dst) and (rank not in self.test_ranks):
                    self.dst.remove(rank)
    
    def run(self) -> None:
        logging.critical(f"Rank {self.rank}\t SRC: {self.src}\t DST: {self.dst}")

        os.environ["MASTER_ADDR"] = self.MASTER_ADDR
        os.environ["MASTER_PORT"] = self.MASTER_PORT
        dist.init_process_group(backend="gloo", rank=self.rank, world_size=self.size)
        accs = []
        for epoch in range(self.epochs):
            if self.rank in self.test_ranks:
                acc = meta_test(
                    self.meta_model,
                    self._test_loader,
                    self.device_id,
                )
                logging.critical(f"Epoch {epoch}\tRank {dist.get_rank()}\tAcc {acc}")
                accs.append(acc)
            epoch_loss = 0
            for data, target in self._train_loader:
                data = TO_CUDA(Variable(data), self.device_id)
                target = TO_CUDA(Variable(target), self.device_id)
                predict_y = TO_CUDA(self.meta_model, self.device_id)(data)
                loss = self.criterion(predict_y, target)

                self.optimizer.zero_grad()
                epoch_loss += loss.item()

                del data, predict_y

                params_dict = get_meta_model_params_dict(self.meta_model)
                classes_params_dict = get_meta_model_classes_params_dict(self.meta_model, labels=[i for i in range(label_len[self.dataset])])
                params = get_meta_model_flat_params(self.meta_model).cpu()
                params_list = [torch.zeros_like(params) for _ in range(len(self.src))]
                grad = collect_grads(self.meta_model, loss).cpu()
                grad_list = [torch.zeros_like(grad) for _ in range(len(self.src))]
                reqs = []
                grad_reqs = []
                if self.attack is not None:
                    for i,s in enumerate(self.src):
                        dist.recv(tensor=params_list[i], src=s)
                        dist.recv(tensor=grad_list[i], src=s)
                    params = self.attack.attack(params_list)
                else:
                    for i,s in enumerate(self.src):
                        req2 = dist.irecv(tensor=params_list[i], src=s)
                        req3 = dist.irecv(tensor=grad_list[i], src=s)
                        reqs.append(req2)
                        grad_reqs.append(req3)
                for d in self.dst:
                    req1 = dist.isend(tensor=params, dst=d)
                    req4 = dist.isend(tensor=grad, dst=d)
                    reqs.append(req1)
                    grad_reqs.append(req4)
                for req in reqs:
                    req.wait()
                for grad_req in grad_reqs:
                    grad_req.wait()

                # aggregate params
                if self.attack is None:
                    params = self.par.par(
                        params,
                        params_list,
                        params_dict,
                        classes_params_dict,
                        2,
                        self.meta_model,
                        self._test_loader,
                        grad,
                        grad_list,
                        self.num_byzantine,
                        self.device_id,
                        self.dataset,
                        target,
                        epoch
                    )
                    del params_list, params_dict, classes_params_dict, grad_list, target
                    set_meta_model_flat_params(self.meta_model, params)
                    set_grads(
                        TO_CUDA(self.meta_model, self.device_id),
                        TO_CUDA(grad, self.device_id),
                    )
                    del params, grad
                    self.optimizer.step()
            logging.critical(
                f"Rank {dist.get_rank()}\tEpoch {epoch}\tTotal Loss {epoch_loss}"
            )
            check_dir(self.logdir)
            if self.attack is None:
                t = np.array(accs)
                np.savetxt(f"{self.logdir}/acc_{self.rank}.csv", t, delimiter=",")

    def set_par(self, par):
        self.par = par
