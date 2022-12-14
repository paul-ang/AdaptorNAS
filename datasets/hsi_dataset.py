import numpy as np
from numpy import loadtxt
import torch
from PIL import Image
from torch.utils.data import Dataset

class HSIDataset(Dataset):
    def __init__(self, root_dir, txt_files, classes=[0, 1, 2, 3, 4],
                 n_cutoff_imgs=None, conv_dims=2):
        """
        :param root_dir: root directory to the dataset folder, e.g ../02-Data/UOW-HSI/
        :param txt_files: text files contain filenames of BMP (ground-truth) files
                               The hsi file has the same filename but with the extension '.raw'
        :param classes: list of classes in the segmentation problem (classes must start from 0)
                        e.g. [0, 1, 2, 3, 4, 5]
        :param n_dims: dimensions of the convolutions
                       + n_dims = 2, return x with a size of (n_bands, H, W).
                       + n_dims = 3, return x with a size of (1, n_bands, H, W).
                                     (# Add a new axis in the first dimension of the original x)
        :param n_cutoff_imgs: maximum number of used images in each text file
        """
        super(HSIDataset, self).__init__()
        self.classes = classes
        self.txt_files = txt_files
        self.root_dir = root_dir
        self.conv_dims = conv_dims

        if (not isinstance(n_cutoff_imgs, int)) or (n_cutoff_imgs <= 0):
            n_cutoff_imgs = None

        # Get filenames of the training images stored in txt_files
        if isinstance(txt_files, str):
            txt_files = [txt_files]
        self.training_imgs = []
        for file in txt_files:
            imgs = list(loadtxt(file, dtype=np.str)[:n_cutoff_imgs])

            if len(self.training_imgs) == 0:
                self.training_imgs = imgs
            else:
                self.training_imgs = np.concatenate((self.training_imgs, imgs))

    def __len__(self):
        """
        :return: the size of the dataset, i.e. the number of hsi images
        """
        return len(self.training_imgs)

    def __getitem__(self, index):
        """
        Read the an image from the training images
        :param index: index of the image in the list of training images
        :return: hsi image and ground-truth images
                + x: input hsi image of size (n_bands, H, W)
                + y_seg: ground-truth segmentation image of size (H, W)
                + y_oht: one-hot coding of the ground-truth image of size (n_classes, H, W)
        """
        # Get the ground truth and raw files
        bmp_file = self.root_dir + '/' + self.training_imgs[index]
        raw_file = self.root_dir + '/' + self.training_imgs[index][:-4] + '.raw'

        # Read the hsi image
        x, _ = hsi_read_data(raw_file)              # of size (H, W, n_bands)
        x = np.moveaxis(x, [0, 1, 2], [1, 2, 0])    # of size (n_bands, H, W)
        x = np.float32(x)                           # convert the input data into float32 datatype

        # Minmax normalization so x in range of [0,1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))

        if self.conv_dims == 3:    # Add a new axis in the first dimension
            x = x[np.newaxis, ...]                  # of size (1, n_bands, H, W)

        # Read the ground-truth image
        bmp = Image.open(bmp_file)
        y_seg = np.array(bmp.getdata()).reshape(bmp.size[1], bmp.size[0])   # of size (H, W)

        # Convert the images into Pytorch tensors
        x = torch.Tensor(x)                                 # of size (n_bands, H, W)
        y_seg = torch.as_tensor(y_seg, dtype=torch.long)    # of size (H, W)

        return {'raw': x, 'label': y_seg}


def hsi_read_data(file_name, sorted=True):
    """
    Read the image cube from the raw hyperspectral image file (.raw)
    :param file_name: file path of the raw hsi file
    :param sorted: bool value to sort the image cube in the ascending of wave_lengths
    :return: 2 params
        - img: image cube in the shape of [n_lines, n_samples, n_bands]
        - wave_lengths: list of wave lengths used to acquired the data
    """
    # Get the information from the header file
    hdr_file = file_name[:-4] + '.hdr'
    n_samples, n_lines, n_bands, data_type, wave_lengths = hsi_read_header(hdr_file)

    # Open the raw file
    f = open(file_name, 'rb')

    # Read the data in the raw file
    data = np.frombuffer(f.read(), dtype=data_type)

    # Close the file
    f.close()

    # Reshape the data into the 3D formar of lines x bands x samples]
    img = data.reshape([n_lines, n_bands, n_samples])

    # Permute the image into the correct format lines x samples x bands
    img = np.moveaxis(img, [0, 1, 2], [0, 2, 1])

    if sorted:
        # Get the sorted indices of wave_lengths in the ascending order
        indx = np.argsort(wave_lengths)

        # Get the sorted wave_lengths
        wave_lengths = wave_lengths[indx]

        # Sort the image cube in the ascending order of wave_lengths
        img = img[:, :, indx]

    return img, wave_lengths

def hsi_read_header(file_name, verbose=False):
    """
    Load the information of a hyperspectral image from a header file (.hdr)
    :param file_name: file path of the header hsi file
    :param verbose: bool value to display the result (defaul: False)
    :return: 5 params
        - n_samples: number of n_samples in the image (width)
        - lines: number of lines in the image (height)
        - bands: number of bands (wave lengths)
        - data_type: data type stored in the data file
        - wave_lengths: list of wave lengths used to acquired the data
    """
    # Open a file for reading
    f = open(file_name, 'r+')

    # Read all the lines in the header file
    text = f.readlines()

    # Close the file
    f.close()

    n = 0
    while n < len(text):
        text_line = text[n].replace('\n', '')
        # Get number of samples (width)
        if 'samples' in text_line:
            n_samples = int(text_line.split(' ')[-1])

        # Get number of lines (height)
        if 'lines' in text_line:
            n_lines = int(text_line.split(' ')[-1])

        # Get number of bands/wave lengths
        if 'bands' in text_line and not(' bands' in text_line):
            n_bands = int(text_line.split(' ')[-1])

        # Get the data type
        if 'data type' in text_line:
            data_type = int(text_line.split(' ')[-1])

        # Get the wave length values
        if 'Wavelength' in text_line:
            wave_lengths = np.zeros(n_bands)
            for k in range(n_bands):
                n = n + 1
                text_line = text[n].replace(',\n', '').replace(' ', '')
                wave_lengths[k] = float(text_line)
            break
        n = n + 1

    # Convert the data_type into the string format
    if data_type == 1:
        data_type = 'uint8'
    elif data_type == 2:
        data_type = 'int16'
    elif data_type == 3:
        data_type = 'int32'
    elif data_type == 4:
        data_type = 'float32'
    elif data_type == 5:
        data_type = 'double'
    elif data_type == 12:
        data_type = 'uint16'
    elif data_type == 13:
        data_type = 'uint32'
    elif data_type == 14:
        data_type = 'int64'
    elif data_type == 15:
        data_type = 'uint64'

    if verbose:     # display the outputs if it is necessary
        print('Image width = %d' % n_samples)
        print('Image height = %d' % n_lines)
        print('Bands = %d' % n_bands)
        print('Data type = %s' % data_type)

    return n_samples, n_lines, n_bands, data_type, wave_lengths


def convert_seg2onehot(y_seg, classes):
    """
    Convert the segmentation image y into the one-hot code image
    :param y_seg: 2D array, a segmentation image with a size of H x W
    :param classes: list of classes in image y
    :return: one-hot code image of y with a size of n_classes x H x W
    """
    y_onehot = np.zeros((len(classes), y_seg.shape[0], y_seg.shape[1]))

    for k, class_label in enumerate(classes):
        y_onehot[k, :, :][y_seg == class_label] = 1

    return y_onehot


def convert_prob2seg(y_prob, classes):
    """
    Convert the class-probability image into the segmentation image
    :param y_prob: class-probability image with a size of n_classes x H x W
    :param classes: list of classes in image y
    :return: 2D array, a segmentation image with a size of H x W
    """
    y_class = np.argmax(y_prob, axis=0)
    y_seg = np.zeros((y_prob.shape[1], y_prob.shape[2]))

    for k, class_label in enumerate(classes):
        # Find indices in y_class whose pixels are k
        indx = np.where(y_class == k)
        if len(indx[0] > 0):
            y_seg[indx] = class_label

    return y_seg



