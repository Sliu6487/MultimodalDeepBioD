import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms


def get_data(data_folder_path=None, up_sample=True):
    print('Importing data...')
    if data_folder_path is None:
        data_folder_path = 'DeepBioD_Dataset/'
    X_train_img = np.load(data_folder_path + 'X_train_img.npy')
    # print("X_train_img has nan:", np.sum(np.isnan(X_train_img)))
    # pytorch channel is at axis 1
    X_train_img = np.rollaxis(X_train_img, 3, 1)
    X_train_tbl = np.load(data_folder_path + 'X_train_descpt.npy')
    # print("X_train_tbl has nan:", np.sum(np.isnan(X_train_tbl)))
    y_train = np.load(data_folder_path + 'y_train.npy')
    # print("y_train has nan:", np.sum(np.isnan(y_train)))

    # make tensors and change types
    X_train_img = torch.tensor(X_train_img).float()
    # replace pos infinite by second largest number
    if torch.sum(torch.isinf(X_train_img)) != 0:
        X_train_img[X_train_img == float("Inf")] = 83
    X_train_tbl = torch.tensor(X_train_tbl).float()
    y_train = torch.tensor(y_train).float()

    torch.manual_seed(50)
    permutation = torch.randperm(y_train.size()[0])
    val_indices = permutation[:218]
    train_indices = permutation[218:]

    X_tr_img = X_train_img[train_indices]
    X_tr_tbl = X_train_tbl[train_indices]
    y_tr = y_train[train_indices]
    X_val_img = X_train_img[val_indices]
    X_val_tbl = X_train_tbl[val_indices]
    y_val = y_train[val_indices]

    if up_sample:
        # todo: stratified sample to the same amount by difference
        # current:   positive: 281; negative: 556
        # up sample: positive: 562; negative: 556
        pos_index = (y_tr == 1).nonzero(as_tuple=True)[0]
        X_tr_img = torch.cat((X_tr_img[pos_index], X_tr_img), 0)
        X_tr_tbl = torch.cat((X_tr_tbl[pos_index], X_tr_tbl), 0)
        y_tr = torch.cat((y_tr[pos_index], y_tr), 0)

    return X_tr_img, X_val_img, X_tr_tbl, X_val_tbl, y_tr, y_val


def create_data_loader(model_number,
                       X_train_tuple,
                       X_val_tuple,
                       y_train, y_val,
                       transform=True,
                       batch_size=32):
    """
    Input tensor data and output data loaders to device
    :param model_number: 3: 'multi', 2: 'table', 1: 'image'
    """
    if (model_number != 2) & transform:
        # TODO: Check if transforms.ToPILImage() is better
        # some version of pytorch only transforms  PIL Image
        # transform = T.RandomRotation(degrees=(0, 180)) this
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor()])
    else:
        transform = None

    if model_number != 3:
        mode_index = model_number - 1
        train_dataset = CustomTensorDataset(tensors=(X_train_tuple[mode_index], y_train),
                                            transform=transform)
        val_dataset = TensorDataset(X_val_tuple[mode_index], y_val)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=128,
                                shuffle=False)

    else:
        train_dataset1 = CustomTensorDataset(tensors=(X_train_tuple[0], y_train),
                                             transform=transform)
        train_dataset2 = CustomTensorDataset(tensors=(X_train_tuple[1], y_train))

        val_dataset1 = TensorDataset(X_val_tuple[0], y_val)
        val_dataset2 = TensorDataset(X_val_tuple[1], y_val)

        train_loader = DataLoader(ConcatDataset(train_dataset1, train_dataset2),
                                  batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(ConcatDataset(val_dataset1, val_dataset2),
                                batch_size=128, shuffle=False)

    return train_loader, val_loader


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        # assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
