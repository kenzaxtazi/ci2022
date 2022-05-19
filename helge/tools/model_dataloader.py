import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from typing import Optional

###############################
#     LightningDataModule     #
###############################

class CI2022_DataModule(pl.LightningDataModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        self.batch_size = hparams.batch_size
        self.num_train = int(hparams.num_traindata)
        self.num_val = int(hparams.num_validata)
        self.num_test = int(hparams.num_testdata)
        self.num_workers = int(hparams.num_workers)
        
        self.data_path_train = hparams.data_path_train
        self.data_path_test = hparams.data_path_test
        self.data_meanstddev = hparams.data_path_meanstddev

        # open xarray dataset
        HDF5_USE_FILE_LOCKING = False
        import xarray as xr
        data = xr.load_dataset(self.data_path_train, chunks={'sample_dim':5000000})
        self.dim_input = data.dims['input_dim']-1
        data.close()
        self.dim_output = 1

        # load mean and stddev
        import xarray as xr
        xr_dataset = xr.open_dataset(self.data_meanstddev)
        self.mean = xr_dataset['mean'].values.tolist()
        self.std = xr_dataset['stddev'].values.tolist()

    def setup(self, stage: Optional[str] = None):
        import xarray as xr

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # Note: in our case we load the pickle-file and create a tensor dataset
            # that is stored within the instance of the Lightning module
            # Train / Validation

            # open xarray dataset
            HDF5_USE_FILE_LOCKING = False
            data = xr.load_dataset(self.data_path_train, chunks={'sample_dim':5000000})

            data = data['samples']

            # normalization
            data = (data - self.mean) / self.std

            X = data.sel(input_dim=slice(0, self.dim_input-1))
            y = data.sel(input_dim=slice(self.dim_input, self.dim_input+self.dim_output))
            print(X.shape)
            print(self.dim_input)
            # create TensorDataset
            tensor_x = torch.Tensor(X.values)
            tensor_y = torch.Tensor(y.values)
            tensor_ds = TensorDataset(tensor_x, tensor_y)   

            # Split and subset
            self.num_train = len(tensor_ds) - self.num_val
            data_train, data_val = random_split(tensor_ds, [self.num_train, self.num_val])
            # assign to use in dataloaders
            self.data_train = data_train
            self.data_val = data_val
            print("Data loaded (train {:.2e}, val {:.2e})".format(self.num_train, self.num_val))

        # Assign test dataset for use in dataloader(s)
        elif stage == 'test' or stage == 'predict':
            # open xarray dataset
            HDF5_USE_FILE_LOCKING = False
            data = xr.load_dataset(self.data_path_test, chunks={'sample_dim':5000000})

            data = data['samples']

            # normalization
            data = (data - self.mean) / self.std

            X = data.sel(input_dim=slice(0, self.dim_input-1))
            y = data.sel(input_dim=slice(self.dim_input, self.dim_input+self.dim_output))

            # create TensorDataset
            tensor_x = torch.Tensor(X.values)
            tensor_y = torch.Tensor(y.values)
            tensor_ds = TensorDataset(tensor_x, tensor_y)

            # assign to use in dataloaders
            self.data_test = tensor_ds
            print("Data loaded (test {:.2e})".format(len(data)))

    def train_dataloader(self):
        """train set removes a subset to use for validation"""
        loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """val set uses a subset of the training set for validation"""
        loader = DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader        

    def test_dataloader(self):
        """test set uses the test split"""
        loader = DataLoader(
            self.data_test,
            batch_size=1000000,#self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        print(loader)
        return loader

    def predict_dataloader(self):
        """test set uses the test split"""
        loader = DataLoader(
            self.data_test,
            batch_size=len(self.data_test),
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader
