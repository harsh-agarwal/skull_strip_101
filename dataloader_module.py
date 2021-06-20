import os

import numpy as np
import pytorch_lightning as pl
from torch import Tensor, unsqueeze
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torchvision import transforms

from utils import get_numpy_from_niigz, resample_nib_image


class DataLoaderModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", split_=0.9):
        super().__init__()
        self.data_dir = data_dir
        self.split = split_
        self.transform = transforms.Compose([transforms.ToTensor()])

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        list_of_folders = os.listdir(os.path.join(self.data_dir))
        list_of_folders.sort()
        # num_samples = len(list_of_folders)
        num_samples = 10
        train_split = [0, int(self.split * 0.9 * num_samples)]
        val_split = [int(self.split * 0.9 * num_samples), int(self.split * num_samples)]
        test_split = [int(self.split * num_samples), num_samples]

        if stage == "fit" or stage is None:

            input_train_data_numpy = np.zeros((0, 128, 128, 96), dtype=np.float32)
            output_train_data_numpy = np.zeros((0, 128, 128, 96), dtype=np.float32)

            for i in range(train_split[0], train_split[1] + 1):

                # load the input image and put it in the input image array
                input_image_path = os.path.join(
                    self.data_dir,
                    list_of_folders[i],
                    "sub-" + list_of_folders[i] + "_ses-NFB3_T1w.nii.gz",
                )

                input_image_data = get_numpy_from_niigz(
                    input_image_path, (128, 128, 96)
                )
                input_train_data_numpy = np.concatenate(
                    (input_train_data_numpy, input_image_data), axis=0
                )

                # load the output image and put in the output image array

                output_image_path = os.path.join(
                    self.data_dir,
                    list_of_folders[i],
                    "sub-" + list_of_folders[i] + "_ses-NFB3_T1w_brain.nii.gz",
                )
                output_image_data = get_numpy_from_niigz(
                    output_image_path, (128, 128, 96)
                )
                output_train_data_numpy = np.concatenate(
                    (output_train_data_numpy, output_image_data), axis=0
                )

            # create a pytorch dataset as a tuple
            self.train_data = TensorDataset(
                unsqueeze(Tensor(input_train_data_numpy), 1),
                unsqueeze(Tensor(output_train_data_numpy), 1),
            )

            input_val_data_numpy = np.zeros((0, 128, 128, 96), dtype=np.float32)
            output_val_data_numpy = np.zeros((0, 128, 128, 96), dtype=np.float32)

            for i in range(val_split[0], val_split[1] + 1):

                # load the input image and put it in the input image array
                input_image_path = os.path.join(
                    self.data_dir,
                    list_of_folders[i],
                    "sub-" + list_of_folders[i] + "_ses-NFB3_T1w.nii.gz",
                )
                input_image_data = get_numpy_from_niigz(
                    input_image_path, (128, 128, 96)
                )
                input_val_data_numpy = np.concatenate(
                    (input_val_data_numpy, input_image_data), axis=0
                )

                # load the output image and put in the output image array

                output_image_path = os.path.join(
                    self.data_dir,
                    list_of_folders[i],
                    "sub-" + list_of_folders[i] + "_ses-NFB3_T1w_brain.nii.gz",
                )
                output_image_data = get_numpy_from_niigz(
                    output_image_path, (128, 128, 96)
                )
                output_val_data_numpy = np.concatenate(
                    (output_val_data_numpy, output_image_data), axis=0
                )

            self.val_data = TensorDataset(
                unsqueeze(Tensor(input_val_data_numpy), 1),
                unsqueeze(Tensor(output_val_data_numpy), 1),
            )

        if stage == "test" or stage is None:

            input_test_data_numpy = np.zeros((0, 128, 128, 96), dtype=np.float32)
            output_test_data_numpy = np.zeros((0, 128, 128, 96), dtype=np.float32)

            for i in range(test_split[0], test_split[1] + 1):

                # load the input image and put it in the input image array
                input_image_path = os.path.join(
                    self.data_dir,
                    list_of_folders[i],
                    "sub-" + list_of_folders[i] + "_ses-NFB3_T1w.nii.gz",
                )
                input_image_data = get_numpy_from_niigz(
                    input_image_path, (128, 128, 96)
                )
                input_test_data_numpy = np.concatenate(
                    (input_test_data_numpy, input_image_data), axis=0
                )

                # load the output image and put in the output image array

                output_image_path = os.path.join(
                    self.data_dir,
                    list_of_folders[i],
                    "sub-" + list_of_folders[i] + "_ses-NFB3_T1w_brain.nii.gz",
                )
                output_image_data = get_numpy_from_niigz(
                    output_image_path, (128, 128, 96)
                )
                output_test_data_numpy = np.concatenate(
                    (output_test_data_numpy, output_image_data), axis=0
                )

            self.test_data = TensorDataset(
                unsqueeze(Tensor(input_test_data_numpy), 1),
                unsqueeze(Tensor(output_test_data_numpy), 1),
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0)
