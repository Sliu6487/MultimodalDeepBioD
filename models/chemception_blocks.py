import torch
import torch.nn as nn


class Stem(nn.Module):
    """
    Stem Layer as defined in https://arxiv.org/abs/1710.02238. The structure is
    significantly altered from the original Inception-ResNet architecture,
    (https://arxiv.org/abs/1602.07261) but the idea behind this layer is to
    downsample the image as a preprocessing step for the Inception-ResNet layers,
    and reduce computational complexity.
    """

    def __init__(self, num_filters, in_channels=4, **kwargs):
        super(Stem, self).__init__(**kwargs)
        """
        Parameters
        ----------
        num_filters: int,
            Number of convolutional filters
        """
        self.num_filters = num_filters
        self.in_channels = in_channels
        """Builds the layers components and set _layers attribute."""

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_filters,
                kernel_size=(4, 4),
                stride=2),
            nn.ReLU()
        )

    def forward(self, x):
        # print("stem input", x.shape)
        out = self.stem(x)
        return out


class InceptionResnetA(nn.Module):
    """
    Variant A of the three InceptionResNet layers described in
    https://arxiv.org/abs/1710.02238. All variants use multiple
    convolutional blocks with varying kernel sizes and number of filters. This
    allows capturing patterns over different scales in the inputs. Residual
    connections are additionally used and have been shown previously to improve
    convergence and training in deep networks. A 1x1 convolution is used on the
    concatenated feature maps from the different convolutional blocks, to ensure
    shapes of inputs and feature maps are same for the residual connection.
    """

    def __init__(self, num_filters, input_channels, **kwargs):
        super(InceptionResnetA, self).__init__(**kwargs)
        """
        Parameters
        ----------
        num_filters: int,
            Number of convolutional filters
        input_channels: int,
            Number of channels in the input.
        """
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.concat_in_channels = self.num_filters + self.num_filters + 2 * self.num_filters

        """Builds the layers components and set _layers attribute."""
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=self.num_filters,
                kernel_size=(3, 3),
                stride=1,
                padding="same"),

            nn.ReLU()

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(1.5 * self.num_filters),
                kernel_size=(3, 3),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=int(1.5 * self.num_filters),
                out_channels=2 * self.num_filters,
                kernel_size=(3, 3),
                stride=1,
                padding="same"),
            nn.ReLU()

        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.concat_in_channels,
                out_channels=self.input_channels,
                kernel_size=(1, 1),
                stride=1,
                padding="same")
        )  # linear activation of final layer

        self.last_reLu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x3 = self.conv_block3(x)
        concat_conv = torch.cat((x1, x2, x3), 1)
        x4 = self.conv_block4(concat_conv)
        sum = x4 + x
        out = self.last_reLu(sum)
        # print('inception out shape', out.shape)
        return out


class InceptionResnetB(nn.Module):
    """
    Variant B of the three InceptionResNet layers described in
    https://arxiv.org/abs/1710.02238. All variants use multiple
    convolutional blocks with varying kernel sizes and number of filters. This
    allows capturing patterns over different scales in the inputs. Residual
    connections are additionally used and have been shown previously to improve
    convergence and training in deep networks. A 1x1 convolution is used on the
    concatenated feature maps from the different convolutional blocks, to ensure
    shapes of inputs and feature maps are same for the residual connection.
    """

    def __init__(self, num_filters, input_channels, **kwargs):
        super(InceptionResnetB, self).__init__(**kwargs)
        """
        Parameters
        ----------
        num_filters: int,
            Number of convolutional filters
        input_channels: int,
            Number of channels in the input.
        """
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.concat_in_channels = self.num_filters + int(self.num_filters * 1.5)

        """Builds the layers components and set _layers attribute."""
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(self.num_filters * 1.25),
                kernel_size=(1, 7),
                stride=1,
                padding="same"),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=int(self.num_filters * 1.25),
                out_channels=int(self.num_filters * 1.5),
                kernel_size=(7, 1),
                stride=1,
                padding="same"),

            nn.ReLU()

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.concat_in_channels,
                out_channels=self.input_channels,
                kernel_size=(1, 1),
                stride=1,
                padding="same")
        )  # linear activation of final layer

        self.last_reLu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        concat_conv = torch.cat((x1, x2), 1)
        x3 = self.conv_block3(concat_conv)
        sum = x3 + x
        out = self.last_reLu(sum)
        return out


class InceptionResnetC(nn.Module):
    """
    Variant C of the three InceptionResNet layers described in
    https://arxiv.org/abs/1710.02238. All variants use multiple
    convolutional blocks with varying kernel sizes and number of filters. This
    allows capturing patterns over different scales in the inputs. Residual
    connections are additionally used and have been shown previously to improve
    convergence and training in deep networks. A 1x1 convolution is used on the
    concatenated feature maps from the different convolutional blocks, to ensure
    shapes of inputs and feature maps are same for the residual connection.
    """

    def __init__(self, num_filters, input_channels, **kwargs):
        super(InceptionResnetC, self).__init__(**kwargs)
        """
        Parameters
        ----------
        num_filters: int,
            Number of convolutional filters
        input_channels: int,
            Number of channels in the input.
        """
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.concat_in_channels = self.num_filters + int(self.num_filters * 1.33)

        """Builds the layers components and set _layers attribute."""
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(self.num_filters * 1.16),
                kernel_size=(1, 3),
                stride=1,
                padding="same"),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=int(self.num_filters * 1.16),
                out_channels=int(self.num_filters * 1.33),
                kernel_size=(3, 1),
                stride=1,
                padding="same"),

            nn.ReLU()

        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.concat_in_channels,
                out_channels=self.input_channels,
                kernel_size=(1, 1),
                stride=1,
                padding="same")
        )  # linear activation of final layer

        self.last_reLu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        concat_conv = torch.cat((x1, x2), 1)
        x3 = self.conv_block3(concat_conv)
        sum = x3 + x
        out = self.last_reLu(sum)
        return out


class ReductionA(nn.Module):
    """
    Variant A of the two Reduction layers described in
    https://arxiv.org/abs/1710.02238. All variants use multiple convolutional
    blocks with varying kernel sizes and number of filters, to reduce the spatial
    extent of the image and reduce computational complexity for downstream layers.
    """

    def __init__(self, num_filters, in_channels, **kwargs):
        super(ReductionA, self).__init__(**kwargs)
        """
        Parameters
        ----------
        num_filters: int,
            Number of convolutional filters
        input_channels: int,
            Number of channels in the input.
        """
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.concat_in_channels = self.num_filters + int(self.num_filters * 1.33)

        """Builds the layers components and set _layers attribute."""

        self.conv_block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=int(self.num_filters * 1.5),
                kernel_size=(3, 3),
                stride=2,
                padding="valid"),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=self.num_filters,
                kernel_size=(3, 3),
                stride=1,
                padding="same"),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(self.num_filters * 1.5),
                kernel_size=(3, 3),
                stride=2,
                padding="valid"),

            nn.ReLU()

        )

        self.last_reLu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x3 = self.conv_block3(x)
        # print("reductA:",x1.shape,x2.shape,x3.shape)
        # concat all channels
        concat_conv = torch.cat((x1, x2, x3), 1)
        # print("reductA after cat:",concat_conv.shape)
        out = self.last_reLu(concat_conv)
        return out


class ReductionB(nn.Module):
    """
    Variant B of the two Reduction layers described in
    https://arxiv.org/abs/1710.02238. All variants use multiple convolutional
    blocks with varying kernel sizes and number of filters, to reduce the spatial
    extent of the image and reduce computational complexity for downstream layers.
    """

    def __init__(self, num_filters, in_channels, **kwargs):
        super(ReductionB, self).__init__(**kwargs)
        """
        Parameters
        ----------
        num_filters: int,
            Number of convolutional filters
        input_channels: int,
            Number of channels in the input.
        """
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.concat_in_channels = self.num_filters + int(self.num_filters * 1.33)

        """Builds the layers components and set _layers attribute."""
        self.conv_block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(self.num_filters * 1.5),
                kernel_size=(3, 3),
                stride=2,
                padding="valid"),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(self.num_filters * 1.125),
                kernel_size=(3, 3),
                stride=2,
                padding="valid"),

            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.num_filters,
                kernel_size=(1, 1),
                stride=1,
                padding="same"),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=self.num_filters,
                out_channels=int(self.num_filters * 1.125),
                kernel_size=(3, 1),
                stride=1,
                padding="same"),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=int(self.num_filters * 1.125),
                out_channels=int(self.num_filters * 1.25),
                kernel_size=(3, 3),
                stride=2,
                padding="valid"),

            nn.ReLU()
        )

        self.last_reLu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x)
        x3 = self.conv_block3(x)
        x4 = self.conv_block4(x)
        # print("reductB:",x1.shape,x2.shape,x3.shape,x4.shape)
        concat_conv = torch.cat((x1, x2, x3, x4), 1)
        out = self.last_reLu(concat_conv)
        return out
