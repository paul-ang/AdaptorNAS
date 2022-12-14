import time
from os import path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

EDGE_COLORS={
    'identity': 'black',
    'conv3': 'green',
    'conv5': 'blue',
    'dconv3': 'pink',
    'dconv3t2': 'purple',
    'conv3t2': 'red',
    'conv3-dil2': 'gold',
    'dconv3-dil2': 'teal'
}

DEFAULT_PRIMITIVES = [
    'identity',
    'conv3',
    'conv3t2',
    'conv3-dil2',
    'zero'
]

CONV3_PRIMITIVES = [
    'identity',
    'conv3',
    'conv3t2',
    'zero'
]

SMALL_PRIMITIVES = [
    'identity',
    'dconv3',
    'dconv3t2',
    'dconv3-dil2',
    'zero'
]



def compute_latency_ms_pytorch(model, input_size, iterations=None, device=None):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size).cuda()

    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in tqdm(range(iterations)):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    # FPS = 1000 / latency (in ms)
    return latency

class BaseOp(nn.Module):
    # Load cached latency as class attribute
    cached_latency_path = "latency_lookup_table.npy"
    if path.isfile(cached_latency_path):
        cached_latency = np.load(cached_latency_path, allow_pickle=True).item()
    else:
        cached_latency = {}

    def __init__(self):
        super(BaseOp, self).__init__()

    def _latency_name(self):
        raise NotImplementedError

    def compute_latency(self, input_size):
        '''
        Args:
            input_size (tuple): (C, H, W)
        Returns:
            latency (ms)
        '''

        name = self._latency_name(input_size[1], input_size[2])
        latency = self._get_latency(name)

        if latency == None:
            print(f"\nComputing latency for {name}")
            latency = compute_latency_ms_pytorch(self, (1, *input_size))
            print(f"{latency} ms")
            self._update_latency(name, latency)
            return latency
        else:
            return latency

    def _get_latency(self, name):
        try:
            latency = BaseOp.cached_latency[name]
            return latency
        except Exception:
            return None

    def _update_latency(self, name, val):
        BaseOp.cached_latency[name] = val
        np.save(BaseOp.cached_latency_path, BaseOp.cached_latency)


class ConvBN(BaseOp):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None,
                 groups=1, bias=False, times=1, dilation=1):
        ''' This creates a convolutional layer , followed by batch norm and ReLU
        Args:
            C_in (int): In channel
            C_out (int): Out channel
            kernel_size (int): kernel size
            stride (int): stride
            padding (int): padding. If None, it will auto-calculate the padding
            so that the output has the same size as the output
            groups (int): channel groups
            bias (bool): include bias param or not?
            times (int): repeats this for n times. E.g. 2 has receptive field of
            a 5x5 conv
            dilation (int): dilation param of the conv
        '''
        super(ConvBN, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.times = times
        self.bias = bias

        layers = nn.ModuleList()
        for i in range(times):
            if i == 0:
                layers.append(nn.Sequential(
                    nn.Conv2d(C_in, C_out, kernel_size, stride,
                              padding='same', bias=bias,
                              dilation=self.dilation),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(C_out, C_out, kernel_size, stride,
                              padding='same', bias=bias,
                              dilation=self.dilation),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
                ))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x

    def _latency_name(self, h, w):
        name = f"ConvBN_H{h}_W{w}_Cin{self.C_in}_Cout{self.C_out}" \
               f"_kernel{self.kernel_size}_stride{self.stride}_groups{self.groups}" \
               f"_times{self.times}_dil{self.dilation}"
        return name


class Zeroize(BaseOp):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x):
        if self.C_in == self.C_out:
            return x.mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

class DepthwiseConv(BaseOp):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None,
                 bias=False, times=1, dilation=1):
        ''' This creates a depthwise-conv layer , followed by batch norm and ReLU
        Args:
            C_in (int): In channel
            C_out (int): Out channel
            kernel_size (int): kernel size
            stride (int): stride
            padding (int): padding. If None, it will auto-calculate the padding
            so that the output has the same size as the output
            bias (bool): include bias param or not?
            times (int): repeats this for n times. E.g. 2 has receptive field of
            a 5x5 conv
        '''
        super(DepthwiseConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.times = times
        self.C_in = C_in
        self.C_out = C_out

        layers = nn.ModuleList()
        for i in range(times):
            if i == 0:
                layers.append(nn.Sequential(
                    nn.Conv2d(C_in, C_in, kernel_size=kernel_size, bias=bias,
                              groups=C_in, stride=stride, padding='same', dilation=dilation),
                    nn.Conv2d(C_in, C_out, kernel_size=1, bias=bias),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(C_out, C_out, kernel_size=kernel_size, bias=bias,
                              groups=C_out, stride=stride,
                              padding='same', dilation=dilation),
                    nn.Conv2d(C_out, C_out, kernel_size=1, bias=bias),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
                ))
            self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x

    def _latency_name(self, h, w):
        name = f"DepthwiseConv_H{h}_W{w}_Cin{self.C_in}_Cout{self.C_out}" \
               f"_kernel{self.kernel_size}_stride{self.stride}" \
               f"_times{self.times}"
        return name


class Skip(BaseOp):
    '''
    Identity function
    '''

    def __init__(self, C_in, C_out):
        '''
        Args:
            C_in(int): the input channel size
            C_out(int): the output channel size
        '''
        super(Skip, self).__init__()
        self.C_in = C_in
        self.C_out = C_out

        if (C_in != C_out):
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = None

    def forward(self, x):
        if self.conv is not None:
            return self.conv(x)
        else:
            return x

    def _latency_name(self, h, w):
        name = f"Skip_H{h}_W{w}_Cin{self.C_in}_Cout{self.C_out}"
        return name


class ResBlock(nn.Module):
    def __init__(self, C, kernel_size=3, padding=None,
                 times=1):
        '''
        Resblock Conv: out = conv(x) + identity
        Args:
            C: in and output channel
            kernel_size: filter size of the conv
            padding: pad the input so the output has same size as input
            times: how many conv?
        '''
        super(ResBlock, self).__init__()
        self.C = C
        self.kernel_size = kernel_size
        self.stride = 1
        self.padding = padding
        self.times = times

        # Compute padding
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(
                np.ceil(((self.kernel_size - 1) + 1 - self.stride) / 2.))
        else:
            self.padding = padding

        layers = nn.ModuleList()
        for i in range(times):
            if i + 1 == times:  # last time or only one repetition
                layers.append(nn.Sequential(
                    nn.Conv2d(self.C, self.C, self.kernel_size, self.stride,
                              padding=self.padding, bias=False),
                    nn.BatchNorm2d(self.C),
                ))
                # Last ReLU is applied after adding the identity
                self.final_relu = nn.ReLU(inplace=True)
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(self.C, self.C, self.kernel_size, self.stride,
                              padding=self.padding, bias=False),
                    nn.BatchNorm2d(self.C),
                    nn.ReLU(inplace=True)
                ))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x = x + identity
        x = self.final_relu(x)

        return x


# A dictionary containing all the operations.
OPS = {
    'identity': lambda C_in, C_out, stride: Skip(C_in, C_out),
    'conv3': lambda C_in, C_out, stride: ConvBN(C_in, C_out, kernel_size=3,
                                                stride=stride),
    'conv3t2': lambda C_in, C_out, stride: ConvBN(C_in, C_out, kernel_size=3,
                                                  stride=stride, times=2),
    'conv5': lambda C_in, C_out, stride: ConvBN(C_in, C_out, kernel_size=5,
                                                stride=stride),
    'dconv3': lambda C_in, C_out, stride: DepthwiseConv(C_in, C_out,
                                                        kernel_size=3,
                                                        stride=stride),
    'dconv3t2': lambda C_in, C_out, stride: DepthwiseConv(C_in,
                                                          C_out,
                                                          kernel_size=3,
                                                          stride=stride,
                                                          times=2),
    'dconv5': lambda C_in, C_out, stride: DepthwiseConv(C_in, C_out,
                                                        kernel_size=5,
                                                        stride=stride),
    'res3t2': lambda C_in, C_out, stride: ResBlock(C_in,
                                                   kernel_size=3,
                                                   times=2),
    'res3': lambda C_in, C_out, stride: ResBlock(C_out,
                                                 kernel_size=3,
                                                 times=1),
    'res5t2': lambda C_in, C_out, stride: ResBlock(C_out,
                                                   kernel_size=5,
                                                   times=2),
    'conv3-dil2': lambda C_in, C_out, stride: ConvBN(C_in, C_out,
                                                     kernel_size=3,
                                                     stride=stride,
                                                     dilation=2),
    'dconv3-dil2': lambda C_in, C_out, stride: DepthwiseConv(C_in, C_out,
                                                             kernel_size=3,
                                                             stride=stride,
                                                             dilation=2),
    'zero': lambda C_in, C_out, stride: Zeroize(C_in, C_out)
}