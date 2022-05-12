import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# Method Siren - Copyright https://github.com/lucidrains/siren-pytorch/


# helpers
def exists(val):
    return val is not None


# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


torch.nn.Sine = Sine


# siren layer
class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0=4., c=6., is_first=False, use_bias=True, activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = torch.nn.Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


torch.nn.Siren = Siren


# Helge Mohn (2021)

class DeepNeuralNet(pl.LightningModule):
    def __init__(self, hparams, w0=4., w0_initial=15., use_bias=True, final_activation=None, *args, **kwargs):
        super().__init__()

        # call this to save params:
        self.save_hyperparameters(hparams)
        self.learning_rate = self.hparams.learning_rate
        self.optimizer = self.hparams.optimizer
        #self.momentum = self.hparams.momentum
        self.dim_input = self.hparams.dim_input
        self.dim_output = self.hparams.dim_output
        self.num_neurons = self.hparams.num_neurons

        # mean/std of ozone tendency should be set in train_siren_xx.ipynb
        self.mean_dox = 0
        self.std_dox = 1

        self.example_input_array = {'x': torch.rand(1, self.dim_input)}

        # =================== Siren Neural Net Architecture ====================
        layers = []

        for ind in range(self.hparams.num_layers-1):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = self.dim_input if is_first else self.num_neurons

            layers.append(Siren(
                dim_in = layer_dim_in, dim_out = self.num_neurons,
                w0 = layer_w0, use_bias = use_bias,
                is_first = is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        layers.append(Siren(
            dim_in = self.num_neurons, dim_out = self.dim_output, 
            w0 = w0, use_bias = use_bias, activation = final_activation))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)

        # 2. Compute loss
        loss = F.mse_loss(y_hat, y)
        #self.log('train_loss', loss, prog_bar=False, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)

        # 2. Compute loss
        loss = F.mse_loss(y_hat, y)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)

        # 2. Compute loss
        return {'test_loss': F.mse_loss(y_hat, y)}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        import numpy as np
        # 1. Forward pass:
        x, y = batch
        #x = x.view(-1, x.size(0))
        y = y.view(y.size(0), -1)
        y_hat = self.forward(x)
        
        y_hat = y_hat * self.std_dox + self.mean_dox
        y = y * self.std_dox + self.mean_dox
        
        y_hat = y_hat.cpu().numpy()
        y = y.cpu().numpy()
        
        loss = (y - y_hat)

        # 2. return prediction
        return {'predict_loss': loss,}
               #'MSE_predict_loss': np.mean(loss**2),
               # 'RMSE_predict_loss': np.sqrt(np.mean(loss)),
               #'std_predict_loss': np.std(np.abs(loss)),
               #'max_predict_loss': np.max(np.abs(loss))}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs])
        self.log('mean_train_loss', avg_loss.mean(), prog_bar=True, logger=True)
        self.log('std_train_loss', avg_loss.std(), prog_bar=True, logger=True)
        self.log('max_train_loss', avg_loss.max(), prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs])
        self.log('mean_val_loss', avg_loss.mean()*self.std_dox, prog_bar=True, logger=True)
        self.log('std_val_loss', avg_loss.std()*self.std_dox, prog_bar=True, logger=True)
        self.log('max_val_loss', avg_loss.max()*self.std_dox, prog_bar=True, logger=True)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs])
        self.log('mean_test_loss', avg_loss.mean()*self.std_dox, logger=True)
        self.log('std_test_loss', avg_loss.std()*self.std_dox, logger=True)
        self.log('max_test_loss', avg_loss.max()*self.std_dox, logger=True)

    def predict_vmr(self, batch, batch_idx):
        x = batch
        y_hat = self.forward(x)
        y_hat = y_hat * self.std_dox + self.mean_dox
        return {'prediction': y_hat}
    
    def configure_optimizers(self):
        # =================== Optimization methods ====================
        optimizers = {
            'Adam': torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate, 
            #    weight_decay=1e-5
            ),
            'AdamW': torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate, 
                betas=(0.9, 0.999), 
                eps=1e-08, 
            #    weight_decay=0.01
            ),
            'LBFGS': torch.optim.LBFGS(
                self.parameters(),
                lr=1, 
                max_iter=20, 
                max_eval=None, 
                tolerance_grad=1e-07, 
                tolerance_change=1e-09, 
                history_size=100),
        }
        optimizer = optimizers[self.optimizer]
        print('using {} optimizer.'.format(self.optimizer))
        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dim_input', type=int, default=52)
        parser.add_argument('--dim_output', type=int, default=1)
        parser.add_argument('--num_layers', type=int, default=6)
        parser.add_argument('--num_neurons', type=int, default=640)#3L 512
        parser.add_argument('--num_traindata', type=int, default=-1)
        parser.add_argument('--num_validata', type=int, default=1000000)#500000)
        parser.add_argument('--num_testdata', type=int, default=-1)#1000000)
        parser.add_argument('--batch_size', type=int, default=100)
        parser.add_argument('--learning_rate', type=float, default=0.00001)
        #parser.add_argument('--num_workers', type=int, default=12)# JUWELS: Dual Intel Xeon Gold 6148 --> 20 cores, 2.4 GH
        return parser