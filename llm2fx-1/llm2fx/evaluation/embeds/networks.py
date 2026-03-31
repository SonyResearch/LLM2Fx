"""
    Implementation of neural networks used in the task 'Music Mastering Style Transfer'
        - 'Effects Encoder'
        - 'Mastering Style Transfer'
        - 'Differentiable Mastering Style Transfer'
"""
import torch
import torch.nn as nn
# compute receptive field
def compute_receptive_field(kernels, strides, dilations):
    rf = 0
    for i in range(len(kernels)):
        rf += rf * strides[i] + (kernels[i]-strides[i]) * dilations[i]
    return rf

# Encoder of music effects for contrastive learning of music effects
class Effects_Encoder(nn.Module):
    def __init__(self, config):
        super(Effects_Encoder, self).__init__()
        # input is stereo channeled audio
        config["channels"].insert(0, 2)

        # encoder layers
        encoder = []
        for i in range(len(config["kernels"])):
            if config["conv_block"]=='res':
                encoder.append(Res_ConvBlock(dimension=1, \
                                                in_channels=config["channels"][i], \
                                                out_channels=config["channels"][i+1], \
                                                kernel_size=config["kernels"][i], \
                                                stride=config["strides"][i], \
                                                padding="SAME", \
                                                dilation=config["dilation"][i], \
                                                norm=config["norm"], \
                                                activation=config["activation"], \
                                                last_activation=config["activation"]))
            elif config["conv_block"]=='conv':
                encoder.append(ConvBlock(dimension=1, \
                                            layer_num=1, \
                                            in_channels=config["channels"][i], \
                                            out_channels=config["channels"][i+1], \
                                            kernel_size=config["kernels"][i], \
                                            stride=config["strides"][i], \
                                            padding="VALID", \
                                            dilation=config["dilation"][i], \
                                            norm=config["norm"], \
                                            activation=config["activation"], \
                                            last_activation=config["activation"], \
                                            mode='conv'))
        self.encoder = nn.Sequential(*encoder)

        # pooling method
        self.glob_pool = nn.AdaptiveAvgPool1d(1)


    # network forward operation
    def forward(self, input):
        enc_output = self.encoder(input)
        glob_pooled = self.glob_pool(enc_output).squeeze(-1)

        # outputs c feature
        return glob_pooled


# 1-dimensional convolutional layer
# in the order of conv -> norm -> activation
class Conv1d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                                    stride=1, \
                                    padding="SAME", dilation=1, bias=True, \
                                    norm="batch", activation="relu", \
                                    mode="conv"):
        super(Conv1d_layer, self).__init__()

        self.conv1d = nn.Sequential()

        ''' padding '''
        if mode=="deconv":
            padding = int(dilation * (kernel_size-1) / 2)
            out_padding = 0 if stride==1 else 1
        elif mode=="conv" or "alias_free" in mode:
            if padding == "SAME":
                pad = int((kernel_size-1) * dilation)
                l_pad = int(pad//2)
                r_pad = pad - l_pad
                padding_area = (l_pad, r_pad)
            elif padding == "VALID":
                padding_area = (0, 0)
            else:
                pass

        ''' convolutional layer '''
        if mode=="deconv":
            self.conv1d.add_module("deconv1d", nn.ConvTranspose1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=padding, output_padding=out_padding, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif mode=="conv":
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
        elif "alias_free" in mode:
            if "up" in mode:
                up_factor = stride * 2
                down_factor = 2
            elif "down" in mode:
                up_factor = 2
                down_factor = stride * 2
            else:
                raise ValueError("choose alias-free method : 'up' or 'down'")
            # procedure : conv -> upsample -> lrelu -> low-pass filter -> downsample
            # the torchaudio.transforms.Resample's default resampling_method is 'sinc_interpolation' which performs low-pass filter during the process
            # details at https://pytorch.org/audio/stable/transforms.html
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=1, padding=0, \
                                                            dilation=dilation, \
                                                            bias=bias))
            self.conv1d.add_module(f"{mode}upsample", torchaudio.transforms.Resample(orig_freq=1, new_freq=up_factor))
            self.conv1d.add_module(f"{mode}lrelu", nn.LeakyReLU())
            self.conv1d.add_module(f"{mode}downsample", torchaudio.transforms.Resample(orig_freq=down_factor, new_freq=1))

        ''' normalization '''
        if norm=="batch":
            self.conv1d.add_module("batch_norm", nn.BatchNorm1d(out_channels))
            # self.conv1d.add_module("batch_norm", nn.SyncBatchNorm(out_channels))

        ''' activation '''
        if 'alias_free' not in mode:
            if activation=="relu":
                self.conv1d.add_module("relu", nn.ReLU())
            elif activation=="lrelu":
                self.conv1d.add_module("lrelu", nn.LeakyReLU())


    def forward(self, input):
        # input shape should be : batch x channel x height x width
        output = self.conv1d(input)
        return output

# Residual Block
    # the input is added after the first convolutional layer, retaining its original channel size
    # therefore, the second convolutional layer's output channel may differ
class Res_ConvBlock(nn.Module):
    def __init__(self, dimension, \
                        in_channels, out_channels, \
                        kernel_size, \
                        stride=1, padding="SAME", \
                        dilation=1, \
                        bias=True, \
                        norm="batch", \
                        activation="relu", last_activation="relu", \
                        mode="conv"):
        super(Res_ConvBlock, self).__init__()

        if dimension==1:
            self.conv1 = Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode)
        elif dimension==2:
            self.conv1 = Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, mode=mode)


    def forward(self, input):
        c1_out = self.conv1(input) + input
        c2_out = self.conv2(c1_out)
        return c2_out
