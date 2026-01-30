import torchvision
from torch.utils.data import random_split, TensorDataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import itertools
import torch
from torch.utils.data import ConcatDataset
import numpy as np
import torch.utils.data as Data
from torch.utils.data import Dataset
import json
import os
from collections import defaultdict


train_transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "CIFAR10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "CIFAR100": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    ),
    "FashionMNIST": transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "FEMNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

test_transforms = {
    "MNIST": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "CIFAR10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "CIFAR100": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    ),
    "FashionMNIST": transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "FEMNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
}

datasets = {
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
}

label_len = {
    "MNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "FashionMNIST": 10,
    "FEMNIST": 62
}
np.random.seed(42)

def split_dataset(dataset_name, workers_n):
    dataset = datasets[dataset_name](
        root="./data",
        train=True,
        download=True,
        transform=train_transforms[dataset_name],
    )
    ave = len(dataset) // workers_n
    lengths = [ave] * (workers_n - 1)
    lengths.append(len(dataset) - ave * (workers_n - 1))
    return random_split(dataset, lengths)


def generate_Truedata(dataset_name):
    sub_datasets = split_dataset(dataset_name, 100)
    valid_loaders = [
        DataLoader(dataset, batch_size=32, shuffle=True)
        for dataset in sub_datasets
    ]
    return valid_loaders[0]


def generate_dataloader(dataset_name, workers_n, batch_size=64):
    sub_datasets = split_dataset(dataset_name, workers_n)
    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in sub_datasets
    ]
    testset = datasets[dataset_name](
        root="./data",
        train=False,
        download=True,
        transform=test_transforms[dataset_name],
    )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loaders, test_loader


def split_dataset_add(dataset_name, workers_n, bate, batch_size=64):
    dataset = datasets[dataset_name](
        root="./data",
        train=True,
        download=True,
        transform=train_transforms[dataset_name],
    )

    sorted_dataset = sorted(dataset, key=lambda x: x[1])
    subset_size = int(len(sorted_dataset) // workers_n)
    subsets_no_iid = [sorted_dataset[i * subset_size: int((i + bate) * subset_size)] for i in range(workers_n)]

    iid_dataset = []
    for i in range(workers_n):
        iid_dataset += sorted_dataset[int((i + bate) * subset_size): (i + 1) * subset_size]

    lengths = [int(subset_size) - int(bate * subset_size)] * (workers_n - 1)
    lengths.append(len(iid_dataset) - sum(lengths))

    subsets_iid = random_split(iid_dataset, lengths)

    sub_dataset = []

    iid = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in subsets_iid
    ]

    no_iid = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in subsets_no_iid
    ]

    for i in range(workers_n):
        sub_dataset.append(itertools.chain(iid[i], no_iid[i]))

    return sub_dataset

def split_dataset_last(dataset_name, workers_n, bate, attacks):
    dataset = datasets[dataset_name](
        root="./data",
        train=True,
        download=True,
        transform=train_transforms[dataset_name],
    )

    sorted_dataset = sorted(dataset, key=lambda x: x[1])
    b=len([i for i in range(workers_n) if attacks[i] is not None])
    subset_size = int(len(sorted_dataset) // (workers_n-b))
    no_iid_num = int(subset_size * bate)
    iid_num = subset_size - no_iid_num
    print(subset_size, "\t", no_iid_num, "\tiid: ", iid_num)

    subsets_no_iid = [sorted_dataset[i * subset_size: int((i + bate) * subset_size)] for i in range(workers_n-b)]

    print("____________________NO_IID_____________")
    for subset in subsets_no_iid:
        class_count = [0] * label_len[dataset_name]
        for _, target in subset:
            class_count[target] += 1
        print(class_count)

    iid_dataset = []
    for i in range(workers_n-b):
        iid_dataset += sorted_dataset[int((i + bate) * subset_size): (i + 1) * subset_size]
    iid_length = len(iid_dataset)
    iid_indices = torch.randperm(iid_length)

    iid_index = []
    for i in range(workers_n - b):
        if i == (workers_n - 1 - b):
            indices = iid_indices[int(i * iid_num):]
        else:
            indices = iid_indices[int(i * iid_num): int((i + 1) * iid_num)]
        iid_index.append(indices)
    subset_iid = []
    for i in range(workers_n - b):
        temp = []
        for j in iid_index[i]:
            temp.append(iid_dataset[j])
        subset_iid.append(temp)

    print("____________________IID_____________")
    for subset in subset_iid:
        class_count = [0] * label_len[dataset_name]
        for _, target in subset:
            class_count[target] += 1
        print(class_count)

    subset_data = []
    count = 0
    b_count = 0
    
    for i in range(workers_n):
        if attacks[i] is None:
            combined_data = subset_iid[count] + subsets_no_iid[count]
            data, target = zip(*combined_data)
            data = torch.stack(data)
            target = torch.tensor(target)  # Convert target to Tensor
            combined_data = TensorDataset(data, target)
            subset_data.append(combined_data)
            count += 1
        elif iid_num != 0:
            b_num = int(subset_size//iid_num)
            b_combined_data = []
            for j in range(b_num):
                b_combined_data += subset_iid[(b_count+b_num) % len(subset_iid)]
            b_data, b_target = zip(*b_combined_data)
            b_data = torch.stack(b_data)
            b_target = torch.tensor(b_target)
            b_combined_data = TensorDataset(b_data, b_target)
            subset_data.append(b_combined_data)
            b_count += b_num
        else:
            b_combined_data = subset_iid[count] + subsets_no_iid[count]
            b_data, b_target = zip(*combined_data)
            b_data = torch.stack(b_data)
            b_target = torch.tensor(b_target)
            b_combined_data = TensorDataset(b_data, b_target)
            subset_data.append(b_combined_data)
            b_count += 1

    return subset_data

def generate_dataloader_add(dataset_name, workers_n, batch_size, bate, attacks):
    if dataset_name=="FEMNIST":
        data_dir = './data/FEMNIST/train/'
        train_loaders = []
        test_loaders = []
        for i in range(workers_n):
            filename = 'all_data_' + str(i) + '_niid_1_keep_0_train_8.json'
            train_path = os.path.join(data_dir, filename)
            test_path = os.path.join(data_dir.replace('train', 'test'), filename.replace('train', 'test'))
            train_dataset = FEMNISTDataset(train_path, transform=train_transforms[dataset_name])
            test_dataset = FEMNISTDataset(test_path, transform=test_transforms[dataset_name])
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        return train_loaders, test_loaders
    
    subset_data = split_dataset_last(dataset_name, workers_n, bate, attacks)
    print("____________________workers_class_count_____________")
    for subset in subset_data:
        class_count = [0] * label_len[dataset_name]
        for _, target in subset:
            class_count[target] += 1
        print(class_count)

    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in subset_data
    ]
    if dataset_name == "HAR":
        test_x = torch.from_numpy(np.load('./data/UCI-HAR/X_test.npy')).float()
        test_y = torch.from_numpy(np.load('./data/UCI-HAR/Y_test.npy')).long()
        test_x = torch.unsqueeze(test_x, 1)
        testset = Data.TensorDataset(test_x, test_y)
    else:
        testset = datasets[dataset_name](
            root="./data",
            train=False,
            download=True,
            transform=test_transforms[dataset_name],
        )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader


def split_dataset_centralized(dataset_name, workers_n, bate, attacks):
    dataset = datasets[dataset_name](
        root="./data",
        train=True,
        download=True,
        transform=train_transforms[dataset_name],
    )

    sorted_dataset = sorted(dataset, key=lambda x: x[1])
    b=len([i for i in range(workers_n) if attacks[i] is not None])
    subset_size = int(len(sorted_dataset) // (workers_n-b-1))
    no_iid_num = int(subset_size * bate)
    iid_num = subset_size - no_iid_num
    print(subset_size, "\t", no_iid_num, "\tiid: ", iid_num)

    subsets_no_iid = [sorted_dataset[i * subset_size: int((i + bate) * subset_size)] for i in range(workers_n-b-1)]

    print("____________________NO_IID_____________")
    for subset in subsets_no_iid:
        class_count = [0] * label_len[dataset_name]
        for _, target in subset:
            class_count[target] += 1
        print(class_count)

    iid_dataset = []
    for i in range(workers_n-b-1):
        iid_dataset += sorted_dataset[int((i + bate) * subset_size): (i + 1) * subset_size]
    iid_length = int(len(sorted_dataset) * (1 - bate))
    iid_indices = torch.randperm(iid_length)

    iid_index = []
    for i in range(workers_n - b - 1):
        if i == (workers_n - 2 - b):
            indices = iid_indices[int(i * iid_num):]
        else:
            indices = iid_indices[int(i * iid_num): int((i + 1) * iid_num)]
        iid_index.append(indices)
    subset_iid = []
    for i in range(workers_n - b - 1):
        temp = []
        for j in iid_index[i]:
            temp.append(iid_dataset[j])
        subset_iid.append(temp)

    print("____________________IID_____________")
    for subset in subset_iid:
        class_count = [0] * label_len[dataset_name]
        for _, target in subset:
            class_count[target] += 1
        print(class_count)

    subset_data = []
    count = 0
    b_count = 0
    
    for i in range(workers_n):
        if attacks[i] is None and i!=0:
            combined_data = subset_iid[count] + subsets_no_iid[count]
            data, target = zip(*combined_data)
            data = torch.stack(data)
            target = torch.tensor(target)  # Convert target to Tensor
            combined_data = TensorDataset(data, target)
            subset_data.append(combined_data)
            count += 1
        elif iid_num != 0 and i!=0:
            b_num = int(subset_size//iid_num)
            b_combined_data = []
            for j in range(b_num):
                b_combined_data += subset_iid[(b_count+b_num) % len(subset_iid)]
            b_data, b_target = zip(*b_combined_data)
            b_data = torch.stack(b_data)
            b_target = torch.tensor(b_target)
            b_combined_data = TensorDataset(b_data, b_target)
            subset_data.append(b_combined_data)
            b_count += b_num
        elif i!=0:
            b_combined_data = subset_iid[count] + subsets_no_iid[count]
            b_data, b_target = zip(*combined_data)
            b_data = torch.stack(b_data)
            b_target = torch.tensor(b_target)  # Convert target to Tensor
            b_combined_data = TensorDataset(b_data, b_target)
            subset_data.append(b_combined_data)
            b_count += 1

    return subset_data

def generate_dataloader_add_centralized(dataset_name, workers_n, batch_size, bate, attacks):
    subset_data = split_dataset_centralized(dataset_name, workers_n, bate, attacks)
    print("____________________workers_class_count_____________")
    for subset in subset_data:
        class_count = [0] * label_len[dataset_name]
        for _, target in subset:
            class_count[target] += 1
        print(class_count)

    train_loaders = [
        DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for dataset in subset_data
    ]
    testset = datasets[dataset_name](
        root="./data",
        train=False,
        download=True,
        transform=test_transforms[dataset_name],
    )
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader


class FEMNISTDataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'r') as f:
            data = json.load(f)
        all_images = []
        all_labels = []
        for user in data['user_data']:
            all_images.extend(data['user_data'][user]['x'])
            all_labels.extend(data['user_data'][user]['y'])
        
        self.images = np.array(all_images, dtype=np.float32).reshape(-1, 28, 28)
        self.labels = np.array(all_labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    from attack import attacks
    from par.bristle import Bristle
    attacks = [None, None, attacks["gaussian"](), None, None, None, None, None, None,None]

    subset = generate_dataloader_add("FashionMNIST",10,64,1.0,attacks)
