#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Block(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        """
        Assembly station for the UNet blocks. 
        The contractive path consists of the repeated application of two 3x3 convolutions 
        (unpadded convolutions), each followed by a rectified linear unit (ReLU) and 
        a 2x2 max pooling operation with stride 2 for downsampling. 
        At each downsampling step we double the number of feature channels.

        :param in_ch: number of input channels
        :param out_ch: number of output channels
        :return: None
        """

        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)


    def forward(self, x):
        """
        Forward pass of the UNet block.

        :param x: input tensor
        :return: output tensor
        """

        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):

    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        """
        Encoder part of the UNet.
        Each block is followed by a 2x2 max pooling operation with stride 2 for downsampling.

        :param chs: number of input channels
        :return: None
        """

        super().__init__()

        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(2)


    def forward(self, x):
        """
        Forward pass of the encoder.

        :param x: input tensor
        :return: list of output tensors
        """

        ftrs = []

        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)

        return ftrs
    

class Decoder(nn.Module):

    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        """
        Decoder part of the UNet.
        Every step in the expansive path consists of an upsampling of the
        feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of
        feature channels, a concatenation with the correspondingly cropped feature map from
        the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is
        necessary due to the loss of border pixels in every convolution.

        :param chs: number of input channels
        :return: None
        """

        super().__init__()

        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])


    def forward(self, x, encoder_features):
        """
        Forward pass of the decoder.

        :param x: input tensor
        :param encoder_features: list of encoder features
        :return: output tensor
        """

        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)

        return x


    def crop(self, enc_ftrs, x):
        """
        Crop the encoder features to the size of the decoder features.

        :param enc_ftrs: encoder features
        :param x: decoder features
        :return: cropped encoder features
        """
        
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)

        return enc_ftrs


class UNet(nn.Module):
    
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3, retain_dim=True, out_sz=(24, 32)):
        """
        UNet model.
        The architecture consists of two parts - the Encoder and the Decoder.

        References:
        - https://arxiv.org/abs/1505.04597
        - https://amaarora.github.io/posts/2020-09-13-unet.html

        :param enc_chs: number of input channels
        :param dec_chs: number of output channels
        :param num_class: number of output classes
        :param retain_dim: retain dimensions of the input image
        :param out_sz: output size of the image
        :return: None
        """
        
        super().__init__()

        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz


    def forward(self, x):
        """
        Forward pass of the UNet.

        :param x: input tensor
        :return: output tensor
        """

        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)

        return out