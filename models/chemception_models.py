import torch
import torch.nn as nn

from models.chemception_blocks import InceptionResnetA, InceptionResnetB, InceptionResnetC
from models.chemception_blocks import Stem, ReductionA, ReductionB


class Chemception(nn.Module):

    def __init__(self,
                 img_spec: str = "engd",
                 img_size: int = 80,
                 base_filters: int = 16,
                 n_classes: int = 1,
                 augment: bool = True,
                 mode: str = "classification",
                 **kwargs):
        super(Chemception, self).__init__()
        # Special attributes
        """
        Parameters
        ----------
        img_spec: str, default std
            Image specification used
        img_size: int, default 80
            Image size used
        base_filters: int, default 16
            Base filters used for the different inception and reduction layers
        inception_blocks: dict,
            Dictionary containing number of blocks for every inception layer
        n_tasks: int, default 10
            Number of classification or regression tasks
        n_classes: int, default 2
            Number of classes (used only for classification)
        augment: bool, default False
            Whether to augment images
        mode: str, default regression
            Whether the model is used for regression or classification
        """
        if img_spec == "engd":
            self.input_shape = (img_size, img_size, 4)
            self.input_channels = 4
        elif img_spec == "std":
            self.input_shape = (img_size, img_size, 1)
        self.base_filters = base_filters
        self.n_classes = n_classes
        self.mode = mode
        self.augment = augment

        # InceptionResnetA
        # InceptionResnetB
        # InceptionResnetC
        # ReductionA
        # ReductionB

        # Modules
        self.stem = Stem(self.base_filters, self.input_channels)
        self.inceptionA = InceptionResnetA(self.base_filters, 16)
        self.reductionA = ReductionA(self.base_filters, 16)
        self.inceptionB = InceptionResnetB(self.base_filters, 64)
        self.reductionB = ReductionB(self.base_filters, 64)
        self.inceptionC = InceptionResnetC(self.base_filters, 126)
        self.avg_poolingC = nn.AvgPool2d((9, 9), 1)
        self.last_linear = nn.Linear(126, n_classes)
        self.sigmoid = nn.Sigmoid()

    def features(self, input):
        # print('input:', input.shape)  # [batch, 4, 80, 80]
        x = self.stem(input)
        # print('stem shape', x.shape)  # [batch, 16, 39, 39]
        x = self.inceptionA(x)
        # print('ceptA out shape', x.shape)  # [batch, 16, 39, 39]
        x = self.reductionA(x)
        # print('reductA out shape', x.shape)  # [batch, 64, 19, 19]
        x = self.inceptionB(x)
        # print('ceptB out shape', x.shape)  # [batch, 64, 19, 19]
        x = self.reductionB(x)
        # print('reductB out shape', x.shape)  # [batch, 126, 9, 9]
        x = self.inceptionC(x)
        # print('inceptC out shape', x.shape)  # [batch, 126, 9, 9]
        return x

    def logits(self, features):
        x = self.avg_poolingC(features)  # [batch, 126, 1, 1]
        # print('global avg_pool out shape', x.shape)
        x = x.view(x.size(0), -1)
        # print('last linear in shape', x.shape)  # [batch,  126]
        x = self.last_linear(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


class Chemception_Small(nn.Module):

    def __init__(self,
                 img_spec: str = "engd",
                 img_size: int = 80,
                 base_filters: int = 16,
                 n_classes: int = 1,
                 augment: bool = True,
                 mode: str = "classification",
                 **kwargs):
        super(Chemception_Small, self).__init__()
        # Special attributs
        """
        Parameters
        ----------
        img_spec: str, default std
            Image specification used
        img_size: int, default 80
            Image size used
        base_filters: int, default 16
            Base filters used for the different inception and reduction layers
        inception_blocks: dict,
            Dictionary containing number of blocks for every inception layer
        n_tasks: int, default 10
            Number of classification or regression tasks
        n_classes: int, default 2
            Number of classes (used only for classification)
        augment: bool, default False
            Whether to augment images
        mode: str, default regression
            Whether the model is used for regression or classification
        """
        if img_spec == "engd":
            self.input_shape = (img_size, img_size, 4)
            self.input_channels = 4
        elif img_spec == "std":
            self.input_shape = (img_size, img_size, 1)
        self.base_filters = base_filters
        self.n_classes = n_classes
        self.mode = mode
        self.augment = augment

        # InceptionResnetA
        # InceptionResnetB
        # InceptionResnetC
        # ReductionA
        # ReductionB

        # Modules
        self.stem = Stem(self.base_filters, self.input_channels)
        self.inceptionA = InceptionResnetA(self.base_filters, 16)
        self.avg_poolingC = nn.AvgPool2d(kernel_size=(39, 39), stride=(1, 1))
        self.last_linear = nn.Linear(16, n_classes)  # [c]

    def features(self, x):
        # print('input:',input.shape) #[4,80,80]
        x = self.stem(x)  # [16, 39, 39]
        # print('stem shape',x.shape)
        x = self.inceptionA(x)  # [16, 39, 39]
        return x

    def logits(self, x):
        x = self.avg_poolingC(x)  # [16, 1, 1]
        # print('global avg_pool out shape',x.shape)
        # print('linear input size:', x.shape)
        x = x.reshape(x.size(0), -1)
        # print('last linear input shape',x.shape)
        x = self.last_linear(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
