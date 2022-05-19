from tools.model_args import get_args
from tools.model_testing import test
from tools.model_dataloader import CI2022_DataModule


###############################
#     Adjust Params here:
###############################
# choose Model:
from tools.model_siren import DeepNeuralNet
exp_name = "drought_prediction_{}L"
exp_description = "CI2022_HACKATHON"
window = -1
ver=1
artifact_dir = './artifacts/model-mtzxw7nb:v0'
###############################



if __name__ == "__main__":
    # ------------
    # hparams
    # ------------

    PATH = artifact_dir + '/model.ckpt'

    from tools.model_siren import DeepNeuralNet
    from tools.model_dataloader import CI2022_DataModule
    from tools.model_args import get_args
    hparams = get_args(DeepNeuralNet, CI2022_DataModule, jupyter_flag=True)

    from pytorch_lightning import seed_everything
    seed_everything(hparams.seed)  

    hparams.gpus = 7
    hparams.window = window
    hparams.version = ver
    hparams.data_path_train = hparams.data_path_train.format(hparams.window, hparams.version)
    hparams.data_path_test = hparams.data_path_test.format(hparams.window, hparams.version)
    hparams.data_path_meanstddev = hparams.data_path_meanstddev.format(hparams.window, hparams.version)
    print(hparams.data_path_train)

    # ------------
    # data
    # ------------
    data = CI2022_DataModule(hparams)
    hparams.dim_input = data.dim_input
    hparams.dim_output = data.dim_output

    # ------------
    # model
    # ------------
    model = DeepNeuralNet(hparams)

    import torch
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']

    model.mean_dox = data.mean[-1]
    model.std_dox = data.std[-1]
    model.eval()


    # ------------
    # ready to start training
    # ------------
    test(model, data, hparams)