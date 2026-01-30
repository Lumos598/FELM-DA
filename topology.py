from data import generate_dataloader_add
from par.par import PAR
from worker import Worker
from models import CIFAR10, MNIST, CIFAR100, FashionMNIST, FEMNIST


class Topology:
    meta_models = {"MNIST": MNIST, "CIFAR10": CIFAR10, "CIFAR100": CIFAR100, "FashionMNIST": FashionMNIST, "FEMNIST": FEMNIST}
    def __init__(self, epochs, logdir, adj_matrix, attacks, par: PAR) -> None:
        self.epochs = epochs
        self.logdir = logdir
        self.par = par
        self.adj_matrix = adj_matrix
        self.attacks = attacks
        assert len(self.adj_matrix) == len(self.attacks)
        self.size = len(self.attacks)
        self.workers = []
        self.non_byzantines = [i for i, _ in enumerate(attacks) if attacks[i] is None]

    def build_topo(self, dataset, batch_size, noniid_rate, args):
        train_loaders, test_loader = generate_dataloader_add(
                dataset, self.size, batch_size=batch_size, bate=noniid_rate, attacks=self.attacks
            )
        # init worker
        if dataset == "FEMNIST":
            meta_model = self.meta_models[dataset]().get_model()
            for rank in range(self.size):
                worker = Worker(
                    rank,
                    self.size,
                    self.attacks[rank],
                    test_ranks=self.non_byzantines,
                    meta_lr=1e-3,
                    train_loader=train_loaders[rank],
                    test_loader=test_loader[rank],
                    model=meta_model,
                    dataset=dataset,
                    epochs=self.epochs,
                    logdir=self.logdir,
                )
                self.workers.append(worker)
        else:
            meta_model = self.meta_models[dataset]()
            for rank in range(self.size):
                worker = Worker(
                    rank,
                    self.size,
                    self.attacks[rank],
                    test_ranks=self.non_byzantines,
                    meta_lr=1e-3,
                    train_loader=train_loaders[rank],
                    test_loader=test_loader,
                    model=meta_model,
                    dataset=dataset,
                    epochs=self.epochs,
                    logdir=self.logdir,
                )
                self.workers.append(worker)
        # build edges
        for i in range(self.size):
            for j in range(self.size):
                if self.adj_matrix[i][j] == 1:
                    self.workers[i].src.append(j)
                    self.workers[j].dst.append(i)
        # set par
        for rank, worker in enumerate(self.workers):
            par = self.par(rank, worker.src, **args)
            worker.set_par(par)
        # set byzantine neighbor number
        for worker in self.workers:
            for i in worker.src:
                worker.num_byzantine += 1 if i not in self.non_byzantines else 0
        # remove or add edges cause byzantine communication
        for worker in self.workers:
            worker.construct_src_and_dst()
