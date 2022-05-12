from argparse import Namespace
from pytorch_lightning import LightningModule, LightningDataModule
import uuid

def train(model: LightningModule, data: LightningDataModule, hparams: Namespace) -> None:
    """
    :param hparams: The arguments to run the model with.
    """
    # ------------
    # logger
    # ------------
    #from pytorch_lightning.loggers import TestTubeLogger
    #import uuid
    #model_dir = '{}/{}'.format(hparams.default_root_dir, str(uuid.uuid4()))
    #print(model_dir)
    #logger = TestTubeLogger(
    #    save_dir=model_dir,
    #    name=hparams.project_name,
    #    description="{}/{}".format(hparams.group_name, hparams.job_type)
    #)

    import wandb
    import os

    #os.environ["WANDB_API_KEY"] = "db9df89354dcf0d7c6a267dc20afdb5a45dc4d37"
    #os.environ["WANDB_MODE"] = "offline"

    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning import Trainer
    wandb_logger = WandbLogger(
        name=hparams.run_name,
        project=hparams.project_name,
        group=hparams.group_name,
        job_type=hparams.job_type,
        log_model=True,#"all",
        save_code=True,
        #offline=True
    )

    # ------------
    # early stopping
    # ------------
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    early_stop_callback = EarlyStopping(
       monitor='mean_val_loss',
       min_delta=0.00,
       patience=30,
       #verbose=True,
       mode='min'
    )

    # ------------
    # model checkpoint
    # ------------
    #from pathlib import Path
    #checkpoint_dir = (
    #    Path(wandb_logger.save_dir)
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        #dirpath=checkpoint_dir,
        #filename='checkpoint-{epoch:02d}-{mean_val_loss:.2e}',
        monitor='mean_val_loss',
        mode='min',
        #save_last=True,
        save_top_k=1
    )

    # ------------
    # training
    # ------------
    from pytorch_lightning import Trainer
    trainer = Trainer.from_argparse_args(
        hparams,# precision=64,
        logger=[wandb_logger],
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    # log gradients and model topology
    wandb_logger.watch(model, log="all", log_freq=1000)
    
    trainer.fit(model, datamodule=data)

    # ------------
    # testing
    # ------------
    if not hparams.fast_dev_run:
        result = trainer.test(ckpt_path='best', datamodule=data)
        from pprint import pprint
        pprint(result)