from tools.model_args import get_args
from tools.model_training import train
from tools.model_dataloader import CI2022_DataModule


###############################
#     Adjust Params here:
###############################
# choose Model:
from tools.model_siren import DeepNeuralNet
exp_name = "drought_prediction_{}L"
exp_description = "CI2022_HACKATHON"

###############################


if __name__ == "__main__":
    # ------------
    # args
    # ------------
    hparams = get_args(DeepNeuralNet, CI2022_DataModule)
    from pytorch_lightning import seed_everything
    seed_everything(hparams.seed)
    
    # Limit Model runtime
    #hparams.max_steps = 200
    hparams.max_epochs = 2
    hparams.val_check_interval = 1000#0.001
    #hparams.max_steps = 10000
    
    hparams.data_path_train = hparams.data_path_train.format(hparams.window, hparams.version)
    hparams.data_path_test = hparams.data_path_test.format(hparams.window, hparams.version)
    hparams.data_path_meanstddev = hparams.data_path_meanstddev.format(hparams.window, hparams.version)
    print(hparams.data_path_train)

    import wandb
    with wandb.init() as run:
        config = wandb.config
        for key in config.keys():
            if key == 'gpus':
                setattr(hparams, key, [config[key]] )
            else:
                setattr(hparams, key, config[key] )
            
    # ------------
    # data
    # ------------
    data = CI2022_DataModule(hparams)
    hparams.dim_input = data.dim_input
    hparams.dim_output = data.dim_output

    # ------------
    # model
    # ------------
    model = DeepNeuralNet(hparams, w0=hparams.w0, w0_initial=hparams.w0_initial)
    print(model)
    model.mean_dox = data.mean[-1]
    model.std_dox = data.std[-1]

    # ------------
    # ready to start training
    # ------------
    train(model, data, hparams)