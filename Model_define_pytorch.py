#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy


# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
n_row = 126
n_col = 128

def frc(H):
    axis0 = np.arange(0,32).astype(np.int32)
    axis0 = np.reshape(axis0, [32])
    axis1 = 4*axis0
    axis2 = 4*axis0+1
    axis3 = 4*axis0+2
    axis4 = 4*axis0+3

    # axis1 = range(0, 127, 4)
    # axis2 = range(1, 127, 4)
    # axis3 = range(2, 127, 4)
    # axis4 = range(3, 127, 4)
    H1 = H[:, :, :, axis1]
    H2 = H[:, :, :, axis2]
    H3 = H[:, :, :, axis3]
    H4 = H[:, :, :, axis4]

    axis5 = np.array([0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]).astype(np.int32)
    axis6 = np.array([4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31]).astype(np.int32)

    H11 = H1[:, :, :, axis5]
    H12 = H1[:, :, :, axis6]
    H21 = H2[:, :, :, axis5]
    H22 = H2[:, :, :, axis6]
    H31 = H3[:, :, :, axis5]
    H32 = H3[:, :, :, axis6]
    H41 = H4[:, :, :, axis5]
    H42 = H4[:, :, :, axis6]

    #result = torch.cat((H11,H12,H21,H22,H31,H32,H41,H42),dim=1)

    return H11,H12,H21,H22,H31,H32,H41,H42


def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


def uQuantize(x, mu):
    #out =torch.sgn(x) * torch.log(1+mu*torch.abs(x))/(1+mu)
    out = torch.log(1+ mu*x)/(1+mu) # input x>=0, ignore the sgn and abs
    return out


def uDequantize(x, mu):
    out = (torch.exp(x*np.log(1+mu))-1)/mu
    return out

class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B, mu):
        ctx.constant = B
        step = 2 ** B
        #out = uQuantize(x, mu)
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B, mu):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        #out = uDequantize(out, mu)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None, None


class DequantizationNonUniform(nn.Module):
    def __init__(self, feedback_bits, B):
        super(DequantizationNonUniform, self).__init__()
        self.B =B
        self.feedback_bits = feedback_bits
        self.dequantizaton = nn.Sequential(
            nn.Linear(feedback_bits, feedback_bits),
            nn.ReLU(),
            nn.Linear(feedback_bits,int(feedback_bits//self.B)),
            nn.ReLU(),
            nn.Linear(int(feedback_bits // self.B), int(feedback_bits // self.B)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.dequantizaton(x)
        return out


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B
        self.mu= 10

    def forward(self, x):
        out = Quantization.apply(x, self.B,self.mu)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B
        self.mu = 10

    def forward(self, x):
        out = Dequantization.apply(x, self.B, self.mu)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv3x1(in_planes, out_planes, stride=1):
    """3x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,1), stride=stride, padding=(1,0), bias=True)


def conv7x1(in_planes, out_planes, stride=1):
    """7x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(7,1), stride=stride, padding=(3,0), bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)


class Encoder(nn.Module):

    def __init__(self, feedback_bits):
        super(Encoder, self).__init__()
        self.B = 4
        self.feedback_bits = feedback_bits
        self.num_ResBlock = 25
        #self.conv1 = conv7x1(4, 4)
        #self.conv2 = conv7x1(4, 4)
        #self.conv3 = conv7x1(4, 4)
        #self.conv4 = conv7x1(4, 4)
        self.multiConvs = nn.ModuleList()
        self.conv_in = nn.Conv2d(2, 16, kernel_size=7, stride=1, padding=3)
        for _ in range(self.num_ResBlock):
            self.multiConvs.append(nn.Sequential(
                conv3x3(16, 32),
                nn.PReLU(),
                conv3x3(32, 16),
                ))
        """
        self.conv_out = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=(2, 1)),  # 16x64x64
            nn.PReLU(),
            #nn.Conv2d(16, 32, kernel_size=(3, 3), stride=4, padding=(2, 1)), # in 16x126x16, out 32x32x4
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=4, padding=1),#in 16x64x64, out 32x16x16
            nn.PReLU(),
            #nn.Conv2d(32, 64, kernel_size=(3, 3), stride=4, padding=(1, 0)),  # out 64x8x1
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=4, padding=1),  #in 32x16x16 out 64x4x4
            nn.PReLU(),
            #nn.Conv2d(64, int(self.feedback_bits // self.B), kernel_size=(3, 3), stride=(8,1), padding=(0, 1)),#out 128x1x1
            nn.Conv2d(64, int(self.feedback_bits // self.B), kernel_size=(3, 3), stride=4, padding=1),
            # in 64x4x4 out 128x1x1

        )
        """
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(8, 32))
        self.fc_in_size = 16*8*32
        self.fc = nn.Linear(self.fc_in_size, int(self.feedback_bits // self.B))
        self.quantize = QuantizationLayer(self.B)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = 10*(x - 0.5)
        #[H11, H12, H21, H22, H31, H32, H41, H42] = frc(out)
        #out1 = self.conv1(torch.cat((H11, H12), dim=1))
        #out2 = self.conv2(torch.cat((H21, H22), dim=1))
        #out3 = self.conv3(torch.cat((H31, H32), dim=1))
        #out4 = self.conv4(torch.cat((H41, H42), dim=1))
        #out = torch.cat((out1,out2,out3,out4),dim=1)
        out = self.conv_in(out)
        out = self.relu1(out)
        for i in range(self.num_ResBlock):
            residual = out
            out = self.multiConvs[i](out)
            out = residual + out
        out = self.relu2(out)
        #out = self.conv_out(out)
        #out = out.view(-1, int(self.feedback_bits // self.B))
        out = self.pooling(out)
        out = out.view(-1, self.fc_in_size)
        out = self.fc(out)
        out = self.sig(out)
        out = self.quantize(out)

        return out


class Decoder(nn.Module):

    def __init__(self, feedback_bits):
        super(Decoder, self).__init__()
        self.B = 4
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        #self.dequantize = DequantizationNonUniform(feedback_bits, self.B)
        self.multiConvs = nn.ModuleList()
        #self.fc = nn.Linear(int(feedback_bits // self.B), n_row*n_col*2)
        self.conv_in = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0),
                nn.PReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),
                nn.PReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=8, stride=8, padding=(1, 0)),
                nn.PReLU(),
        )
        self.out_cov = conv3x3(16, 2)
        self.sig = nn.Sigmoid()
        self.num_ResBlock = 25

        for _ in range(self.num_ResBlock):
            self.multiConvs.append(nn.Sequential(
                conv3x3(16, 32),
                #nn.BatchNorm2d(8),
                nn.PReLU(),
                conv3x3(32, 16),
                #nn.BatchNorm2d(16),
                #nn.PReLU(),
                #conv3x3(16, 2),
                #nn.BatchNorm2d(2),
                ))

    def forward(self, x):
        out = self.dequantize(x)
        out = 10*(out-0.5)
        out = out.view(-1, int(self.feedback_bits // self.B), 1, 1)
        out = self.conv_in(out)
        for i in range(self.num_ResBlock):
            residual = out
            out = self.multiConvs[i](out)
            out = residual + out

        out = self.out_cov(out)
        out = self.sig(out)
        return out

# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]
