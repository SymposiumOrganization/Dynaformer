from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from dynaformer.data_modules.data_modules import*
import hydra
from pathlib import Path
from dynaformer.models import Dynaformer


@hydra.main(config_path="../config/", config_name='train')
def main(cfg):
    """
    Entry point for training.
    """
    data_dir = Path(hydra.utils.to_absolute_path(cfg.data_dir))

    if cfg.method.name == "dynaformer":
        model = Dynaformer(final_out_dim=1,lr=cfg.method.lr, is_instance_norm=cfg.method.instance_norm, loss=cfg.loss, patience_lr_plateau=cfg.patience_lr_plateau)
        save_top_k = 1
        drop_final=True
    else:
        raise Exception("Model type not recognized")

    data = DataModuleExpII(data_dir=data_dir, batch_size=cfg.method.batch_size, num_w=cfg.num_w, requires_normalization=cfg.method.requires_normalization, is_single_query=cfg.method.is_single_query,
                        min_init=cfg.min_init, max_init=cfg.max_init,min_length=cfg.min_length,max_length=cfg.max_length,drop_final=drop_final)   

    checkpoint_callback_pred = ModelCheckpoint(
        monitor='val_loss',
        dirpath='weights/',
        filename=  cfg.method.name + '-{epoch:02d}-{prediction_error:.2f}',
        mode='min',
    )

    checkpoint_callback_train = ModelCheckpoint(
        monitor='train_loss',
        dirpath='weights/',
        filename= cfg.method.name + '_train_'+'-{epoch:02d}-{prediction_error:.2f}',
        mode='min',
        save_top_k=save_top_k,
    )

    trainer = Trainer(
        callbacks=[checkpoint_callback_pred,checkpoint_callback_train],
        #strategy='ddp',            
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=3,
        num_sanity_val_steps=-1,
        accelerator="gpu",
        gpus=cfg.gpu
        
    )   

    trainer.fit(model, data)



if __name__ == "__main__":
    main()