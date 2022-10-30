import os
import piq
import skimage
import statistics

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

NUM_IMGS_TO_TEST = 100


class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.input_range = (0.0, 1.0)
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

        for file in self.files:
            tensor = read_image(os.path.join(
                self.root_dir, file)).double() / 255.
            # tensor = torch.tensor(skimage.io.imread(os.path.join(self.root_dir, file))).permute(
            #     2, 0, 1)[None, ...] / 255.
            # mean, std = tensor.mean([1, 2]), tensor.std([1, 2])
            # transform_norm = transforms.Normalize(mean, std)
            # tensor_norm = transform_norm(tensor)
            self.data.append(tensor)

        self.data = torch.stack(self.data).uniform_(*self.input_range)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x = self.data[index]
        return {'images': x}


def pnsr(gt, gan):
    # To compute PSNR as a measure, use lower case function from the library.
    psnr_index = piq.psnr(gt, gan, data_range=1., reduction='none')
    return psnr_index


def ssim(gt, gan):
    # To compute SSIM index as a measure, use lower case function from the library:
    ssim_index = piq.ssim(gt, gan, data_range=1.)
    # In order to use SSIM as a loss function, use corresponding PyTorch module:
    # ssim_loss: torch.Tensor = piq.SSIMLoss(data_range=1.)(gt, gan)
    return ssim_index  # , ssim_loss


def fsim(gt, gan):
    # To compute FSIM as a measure, use lower case function from the library
    fsim_index: torch.Tensor = piq.fsim(
        gt, gan, data_range=1., reduction='none')
    # In order to use FSIM as a loss function, use corresponding PyTorch module
    # fsim_loss = piq.FSIMLoss(data_range=1., reduction='none')(gt, gan)
    return fsim_index  # , fsim_loss


def msssim(gt, gan):
    # To compute MS-SSIM index as a measure, use lower case function from the library:
    ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(gt, gan, data_range=1.)
    # In order to use MS-SSIM as a loss function, use corresponding PyTorch module:
    # ms_ssim_loss = piq.MultiScaleSSIMLoss(
    #     data_range=1., reduction='none')(gt, gan)
    return ms_ssim_index  # , ms_ssim_loss


def fid(feats_gt, feats_gan):
    # Use FID class to compute FID score from image features, pre-extracted from some feature extractor network
    fid: torch.Tensor = piq.FID()(feats_gt, feats_gan)


@torch.no_grad()
def main():
    path = os.getcwd()
    path_gt = "D:\\DOMI\\University\\Magistrale\\Tesi\\Git_repo\\Real-ESRGAN\\inference\\gt\\"
    path_lq = "D:\\DOMI\\University\\Magistrale\\Tesi\\Git_repo\\Real-ESRGAN\\inference\\lq\\"
    path_gan = "D:\\DOMI\\University\\Magistrale\\Tesi\\Git_repo\\Real-ESRGAN\\inference\\gan\\"

    all_files_gt = os.listdir(path_gt)
    # random_selected_files_gt = random.sample(all_files_gt, NUM_IMGS_TO_TEST)
    all_files_lq = os.listdir(path_lq)
    # random_selected_files_lq = random.sample(all_files_lq, NUM_IMGS_TO_TEST)
    all_files_gan = os.listdir(path_gan)
    # random_selected_files_gan = random.sample(all_files_gan, NUM_IMGS_TO_TEST)

    # metrics_gt = {"img":[{"psnr":0, "ssim":0, "fsim":0, "ms_sim": 0}]}
    # metrics_gan = {"img":[{"psnr":0, "ssim":0, "fsim":0, "ms_sim": 0}]}
    metrics_gt = {"psnr": [], "ssim": [], "fsim": [], "ms_ssim": []}
    metrics_gan = {"psnr": [], "ssim": [], "fsim": [], "ms_ssim": []}

    for file_gt, file_lq, file_gan in zip(all_files_gt, all_files_lq, all_files_gan):
        # Read RGB image and it's noisy version
        gt = torch.tensor(skimage.io.imread(os.path.join(path_gt, file_gt))).permute(
            2, 0, 1)[None, ...] / 255.
        lq = torch.tensor(skimage.io.imread(os.path.join(path_lq, file_lq))).permute(
            2, 0, 1)[None, ...] / 255.
        gan = torch.tensor(skimage.io.imread(os.path.join(
            path_gan, file_gan))).permute(2, 0, 1)[None, ...] / 255.

        # if torch.cuda.is_available():
        #     # Move to GPU to make computaions faster
        #     gt = gt.cuda()
        #     gan = gan.cuda()

        # Image Metrics:
        psnr_gt = pnsr(lq, gt).numpy()[0]
        ssim_gt = float(ssim(lq, gt).numpy())
        fsim_gt = fsim(lq, gt).numpy()[0]
        ms_ssim_gt = float(msssim(lq, gt).numpy())

        metrics_gt["psnr"].append(psnr_gt)
        metrics_gt["ssim"].append(ssim_gt)
        metrics_gt["fsim"].append(fsim_gt)
        metrics_gt["ms_ssim"].append(ms_ssim_gt)

        psnr_gan = pnsr(lq, gan).numpy()[0]
        ssim_gan = float(ssim(lq, gan).numpy())
        fsim_gan = fsim(lq, gan).numpy()[0]
        ms_ssim_gan = float(msssim(lq, gan).numpy())

        metrics_gan["psnr"].append(psnr_gan)
        metrics_gan["ssim"].append(ssim_gan)
        metrics_gan["fsim"].append(fsim_gan)
        metrics_gan["ms_ssim"].append(ms_ssim_gan)

    # stats:
    psnr_mean_gt = statistics.mean(metrics_gt["psnr"])
    ssim_mean_gt = statistics.mean(metrics_gt["ssim"])
    fsim_mean_gt = statistics.mean(metrics_gt["fsim"])
    ms_ssim_mean_gt = statistics.mean(metrics_gt["ms_ssim"])

    print("")
    print("-" * 58)
    print(f"Test:")
    print(
        f"PSNR gt: {psnr_mean_gt:0.2f}, PSNR gan: {psnr_gan:0.2f}")
    print(
        f"SSIM gt: {ssim_mean_gt:0.2f}, SSIM gan: {ssim_gan:0.2f}")
    print(
        f"FSIM gt: {fsim_mean_gt:0.2f}, FSIM gan: {fsim_gan:0.2f}")
    print(
        f"MS-SSIM gt: {ms_ssim_mean_gt:0.2f}, MS_SSIM gan: {ms_ssim_gan:0.2f}")
    print("-" * 58)
    print("")

    dataset_gt = MyImageFolder(
        "D:\\DOMI\\University\\Magistrale\\Tesi\\Git_repo\\Real-ESRGAN\\inference\\gt\\")
    loader_gt = DataLoader(dataset_gt)
    features_gt = piq.FID().compute_feats(loader_gt, device='cpu')

    dataset_gan = MyImageFolder(
        "D:\\DOMI\\University\\Magistrale\\Tesi\\Git_repo\\Real-ESRGAN\\inference\\gan\\")
    loader_gan = DataLoader(dataset_gan)
    features_gan = piq.FID().compute_feats(loader_gan, device='cpu')

    # Use FID class to compute FID score from image features, pre-extracted from some feature extractor network
    fid: torch.Tensor = piq.FID()(features_gt, features_gan)
    print(f"FID: {fid:0.4f}")

    # Use inception_score function to compute IS from image features, pre-extracted from some feature extractor network.
    # Note, that we follow recommendations from paper "A Note on the Inception Score"
    isc_mean, _ = piq.inception_score(features_gt, num_splits=10)
    # To compute difference between IS for 2 sets of image features, use IS class.
    isc: torch.Tensor = piq.IS(distance='l1')(features_gt, features_gan)
    print(f"IS: {isc_mean:0.4f}, difference: {isc:0.4f}")


if __name__ == '__main__':
    main()
