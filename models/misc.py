import numpy as np
from torch import nn


class SizeNormLayer(nn.Module):
    """
    - [SizeNormLayer] is a class that can pad the input N-D tensor so that the size
     in each dimension can be divisible by a divisible_number (default = 16).

    - This class can also crop the input N-D tensor given a padding size.

    Note that the format of the input data: (batch, n_channels, 1st_size, ..., nth_size)
    """

    def __init__(self, mode="pad", divisible_number=16):
        """
        :param mode: "pad" or "crop"
        """
        super(SizeNormLayer, self).__init__()
        self.mode = mode
        self.divisible_number = divisible_number
        self.padding = []

    def forward(self, x, padding=None):
        if self.mode == "pad":  # do padding
            # Compute the padding amount so that each dimension is divisible
            # by divisible_number
            padding = self.compute_padding(x)
            self.padding = padding

            # Apply padding
            x = nn.functional.pad(x, padding)

            return x

        else:  # do cropping
            # Cropping by padding with minus amount of padding
            padding = tuple([-p for p in padding])
            # print('cropping', padding)

            # Apply cropping
            x = nn.functional.pad(x, padding)

            return x

    def compute_padding(self, x):
        """
        Computes the padding for each spatial dim (exclude depth) so that it is
        divisible by divisible_number
        :param x: N-D tensor with the data format
        :param divisible_number:
        :return: padding value at each dimension, e.g. 2D->(d2_p1, d2_p2, d1_p1, d1_p2)
        """
        padding = []
        input_shape_list = x.size()

        # Reversed because pytorch's pad() receive in a reversed order
        for org_size in reversed(input_shape_list[2:]):
            # compute the padding amount in two sides
            p = np.int32((np.int32((org_size - 1) / self.divisible_number) + 1)
                         * self.divisible_number - org_size)
            # padding amount in one size
            p1 = np.int32(p / 2)
            padding.append(p1)
            padding.append(p - p1)

        return tuple(padding)