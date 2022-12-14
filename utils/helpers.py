import os
import time
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils import benchmark
from torch.utils.data import random_split
from tqdm import tqdm
import matplotlib as mpl
from datasets.hsi_dataset import HSIDataset, convert_prob2seg
from sklearn.metrics import confusion_matrix
import numpy as np
import torchmetrics as tm


def compute_metrics(pred_max, label, num_classes=5):
    '''
    :param pred_max: segmentation map (batch, H, W)
    :param label: ground-truth map (batch, H, W)
    :return: acc, iou, dice
    '''

    n_ext_classes = len(torch.unique(torch.stack((pred_max, label), dim=0)))
    macro_iou = tm.functional.iou(pred_max, label, reduction='sum') / n_ext_classes  # macro
    micro_acc = tm.functional.accuracy(pred_max, label, average='micro')  # micro = global accuracy
    macro_acc = tm.functional.accuracy(pred_max, label, average='macro', num_classes=num_classes)  # micro = global accuracy
    macro_dice = tm.functional.f1(pred_max, label, average='macro', num_classes=num_classes, mdmc_average='global')  # macro

    return_dict = {
        'macro_iou': macro_iou,
        'micro_acc': micro_acc,
        'macro_acc': macro_acc,
        'macro_dice': macro_dice
    }

    return return_dict


def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if visual_folder is True:
            os.mkdir(path + '/visual')  # for visual results
    else:
        print("DIR already exists.")
    print('Experiment dir : {}'.format(path))


def get_training_dataloaders(batch_size, num_workers, root_dir, fold: int =1):
    assert fold >= 1 and fold <= 5, "Fold number is invalid!"
    all_txtfiles = [f'{root_dir}/partition/P1.txt',
                    f'{root_dir}/partition/P2.txt',
                    f'{root_dir}/partition/P3.txt',
                    f'{root_dir}/partition/P4.txt',
                    f'{root_dir}/partition/P5.txt']

    # Setup the five-folds
    train_files = []
    for i in range(5):
        if i == (fold-1):
            test_files = all_txtfiles[i]
        else:
            train_files.append(all_txtfiles[i])

    # Check for cross-contamination
    assert any(elem in train_files for elem in test_files) is False, "The train and test sets are contaminated!"

    # Train dataloaders
    full_train_dataset = HSIDataset(root_dir, txt_files=train_files)

    num_train = len(full_train_dataset)
    split = int(np.floor(0.1 * num_train))  # 10% for validation
    train_dataset, val_dataset = random_split(full_train_dataset, [num_train-split, split],
                                      generator=torch.Generator().manual_seed(42))

    assert len(train_dataset) + len(val_dataset) == num_train

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )

    # Validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    # Test dataloader
    test_dataset = HSIDataset(root_dir, txt_files=test_files)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Check if the sum of train, test and val sets are equal to the total number of images
    assert (len(train_loader.sampler) + len(val_loader.sampler) + len(
        test_loader.sampler)) == (len(train_dataset) + len(val_dataset) + len(test_dataset))

    print("Dataset information")
    print("-------------------")
    print(f"Train set: {len(train_loader.sampler)}")
    print(f"Val set: {len(val_loader.sampler)}")
    print(f"Test set: {len(test_loader.sampler)}")
    print(f"Train files: {train_files}")
    print(f"Test files: {test_files}")
    print("-------------------")

    return train_loader, val_loader, test_loader


def create_segmentation_map(pred, raw):
    '''
    For visualization, it overlay pred to raw
    :param pred: the predicted segmentation map
    :param raw: the raw image, which is already normalized by 255.
    :return:
    '''
    num_class = np.unique(pred)  # get num_class from the final output
    pred_map = raw.copy()
    for j in num_class:
        class_idx = (pred == j)  # for j-th class
        if j == 1:  # class 1
            pred_map[class_idx, 0] = 81 / 255
            pred_map[class_idx, 1] = 164 / 255
            pred_map[class_idx, 2] = 82 / 255
        elif j == 2:  # class 2
            pred_map[class_idx, 0] = 204 / 255
            pred_map[class_idx, 1] = 0 / 255
            pred_map[class_idx, 2] = 102 / 255

    # Overlay segmentation on raw img
    alpha = 0.5
    overlayed_pred = cv2.addWeighted(pred_map, alpha, raw, 1 - alpha, 0)

    return overlayed_pred


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

def check_speed_per_operation(op, input_size):
    ''' Test the inference speed of a deep network
    :param model: the deep network
    :param cfg: yml configurations
    :return:  time per image (ms)
    '''
    with torch.no_grad():
        op.eval()
        # Create our test tensor
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        test_tensor = torch.randn(input_size).cuda()
        test_tensor = test_tensor.float()

        torch.cuda.synchronize()
        torch.cuda.synchronize()

        warm_up = 30
        # Warm up
        for i in range(warm_up):
            with torch.no_grad():
                _ = op(test_tensor)
            torch.cuda.synchronize()

        # Measure speed
        total_time = 0.0
        num_test = 1000

        for i in range(num_test):
            inference_start = time.perf_counter()
            with torch.no_grad():
                _ = op(test_tensor)

            torch.cuda.synchronize()
            inference_time_taken = time.perf_counter() - inference_start
            total_time += inference_time_taken

        time_per_image = total_time / num_test * 1000  # convert to ms otherwise gradient is too small
    return time_per_image

def check_model_speed(model, device='cuda', times=100):
    def loop_func(model, x):
        with torch.no_grad():
            x = model(x)
        return x

    x = torch.randn(1, 25, 217, 409).to(device)
    model = model.to(device).eval()

    t0 = benchmark.Timer(stmt="loop_func(model, x)", globals={'model': model, 'x': x, 'loop_func': loop_func})

    return t0.timeit(times).mean


def check_speed(model, cfg):
    ''' Test the inference speed of a deep network
    :param model: the deep network
    :param cfg: yml configurations
    :return:  time per image (s)
    '''
    with torch.no_grad():
        model.eval()
        # Create our test tensor
        torch.cuda.empty_cache()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        input_size = (217, 409)
        tensor_shape = (1, 25,) + input_size
        test_tensor = torch.randn(tensor_shape).to("cuda:0")
        test_tensor = test_tensor.float()

        torch.cuda.synchronize()
        torch.cuda.synchronize()

        warm_up = 30
        # Warm up
        for i in range(warm_up):
            with torch.no_grad():
                output = model(test_tensor)
            torch.cuda.synchronize()

        # Measure speed
        total_time = 0.0
        num_test = 1000

        for i in range(num_test):
            inference_start = time.perf_counter()
            with torch.no_grad():
                output = model(test_tensor)

            torch.cuda.synchronize()
            inference_time_taken = time.perf_counter() - inference_start
            total_time += inference_time_taken

        time_per_image = total_time / num_test
    return time_per_image


def compute_confusion_matrix(y_gt, y_pr, classes=[0, 1, 2, 3, 4]):
    """
    :param y_gt: array of size (batchsize, 1, H, W)
    :param y_pr: array of size (batchsize, n_classes, H, W)
    :param classes: list of class labels
    :return: confusion matrix of the y_gt and segmentation of y_pr
    """
    cm = 0
    for k in range(y_gt.shape[0]):
        # Convert the current y_pr in to the segmentation
        y_prk = convert_prob2seg(np.squeeze(y_pr[k, ...]), classes).flatten()

        # Get the kth ground-truth segmentaion
        y_gtk = np.squeeze(y_gt[k, ...]).flatten()

        # Sum up the confusion matrix
        cm = cm + confusion_matrix(y_gtk, y_prk, labels=classes)

    return cm


def show_visual_results(x, y_gt, y_pr, classes=[0, 1, 2, 3, 4],
                        show_visual=0, comet=None, fig_name=""):
    """
    Show the pseudo-color, ground-truth, and output images
    :param x: array of size (batchsize, n_bands, H, W)
    :param y_gt: array of size (batchsize, 1, H, W)
    :param y_pr: array of size (batchsize, n_classes, H, W)
    :param classes: list of class labels
    :param show_visual: boolean to display the figure or not
    :param comet: comet logger object
    :param fig_name: string as the figure name for the comet logger
    :return:
    """
    # Select the first image in the batch to display
    y_gt = y_gt[0, ...]                         # of size (H, W)
    y_pr = y_pr[0, ...]                         # of size (n_classes, H, W)
    x = np.squeeze(x)                           # of size (n_bands, H, W)
    x = np.moveaxis(x, [0, 1, 2], [2, 0, 1])    # of size (H, W, n_bands)

    # Convert the probability into the segmentation result image
    y_pr = convert_prob2seg(y_pr, classes)

    # Set figure to display
    h = 2
    if plt.fignum_exists(h):  # close if the figure existed
        plt.close()
    fig = plt.figure(figsize=(9, 2))
    fig.subplots_adjust(bottom=0.5)

    # Set colormap
    # colors = ['black', 'green', 'blue', 'red', 'yellow']
    colors = [[0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]]
    cmap_ygt = mpl.colors.ListedColormap(colors[np.int(np.min(y_gt)):np.int(np.max(y_gt)) + 1])
    cmap_ypr = mpl.colors.ListedColormap(colors[np.int(np.min(y_pr)):np.int(np.max(y_pr)) + 1])

    # Plot the pseudo-color image
    plt.subplot(131)
    plt.imshow(hsi_img2rgb(x))
    plt.title('Pseudo-color image')

    # Plot the ground-truth image
    ax = plt.subplot(132)
    im = ax.imshow(y_gt, cmap=cmap_ygt)
    plt.title('Ground-truth image')
    fig.colorbar(im, ax=ax)

    # Plot the predicted segmentation image
    ax = plt.subplot(133)
    im = ax.imshow(y_pr, cmap=cmap_ypr)
    plt.title('Predicted segmentation image')
    fig.colorbar(im, ax=ax)

    if show_visual:
        plt.show()
    return fig

def hsi_img2rgb(img, wave_lengths=None):
    """
    Convert raw hsi image cube into a pseudo-color image
    :param img: 3D array of size H x W x bands
    :param wave_lengths: 1D array of wavelength bands
    :return: array of size H x W x 3, a pseudo-color image
    """
    # Get the indices of ascending-sorted wavelengths
    if wave_lengths is None:
        indx = list(range(img.shape[-1]))
    else:
        indx = np.argsort(wave_lengths)

    # Get the pseudo-red channel (slice of the longest wavelength)
    ind = indx[-1]
    r = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Get the pseudo-green channel (slice of the median wavelength)
    ind = indx[len(indx)//2]
    g = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Get the pseudo-blue channel (slice of the shortest wavelength)
    ind = indx[0]
    b = norm_inten(img[:, :, ind])[..., np.newaxis]

    # Concatenate the channels into a color image
    rgb_img = np.concatenate([r, g, b], axis=-1)

    return rgb_img.astype(np.uint8)


def save_model(model, save_filename):
    """
    Save the weight of model into save_filename
    """
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_filename)
    else:
        torch.save(model.state_dict(), save_filename)


def norm_inten(I, max_val=255):
    """
    Normalize intensities of I to the range of [0, max_val]
    :param I: ndarray
    :param max_val: maximum value of the normalized range, default = 255
    :return: normalized ndarray
    """
    I = I - np.min(I)
    I = (max_val/np.max(I)) * I

    return I
