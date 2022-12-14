# Setup comet-ml logger
from comet_ml import Experiment
import torch
import time
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from models.proposed_method.nas_candidate import NASCandidate
from models.proposed_method.nas_decoder import Supernet
from utils.helpers import create_exp_dir, get_training_dataloaders, \
    check_model_speed


def main():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # Can leave default
    parser.add_argument('--debug', default=0, type=int, help="Debug mode?")
    parser.add_argument('--seed', default=8402, type=int, help="Run the experiment with this random seed.")
    parser.add_argument('--num_workers', default=6, type=int, help="Number of workers for the dataloaders.")
    parser.add_argument('--dataset_dir', default="/projects/datasets/UOW-HSI-v2", type=str, help="The path to the dataset.")
    parser.add_argument('--batch_size', default=4, type=int, help="Batch size for the dataloaders.")
    parser.add_argument('--nas_layers', default=4, type=int, help="The number of layers (L) for the search space.")
    parser.add_argument('--nas_max_edges', default=2, type=int, help="Retain only p outgoing edges for each node.")
    parser.add_argument('--nas_selection_epochs', default=5, type=int, help="Train for n epochs after removal of edges and operations.")
    parser.add_argument('--nas_encoder', default="resnet34", type=str, help="The encoder model for the AdaptorNAS.", choices=["resnet34", "mobilenet_v2", "efficientnet-b2"])
    parser.add_argument('--monitor_loss', default="val_mIOU", type=str, help="Which metric to monitor for the early stopping and the pruning.", choices=['val_acc', 'val_mIOU'])
    parser.add_argument('--nas_weight', type=str, default='', help="Skip the supernet training by loading the saved weight.")
    parser.add_argument('--nas_ops_set', type=str, default='default', help="default or small ops set.")
    parser.add_argument('--lr', default=0.0001, help="The learning rate for the optimizer.", type=float)
    # Required
    parser.add_argument('--name', required=True, help="Name of the experiment.", type=str)
    parser.add_argument('--fold', required=True, help="Use which cross-validaiton fold data?", type=int)
    args = parser.parse_args()

    # Initialize a dummy logger for code compatibility.
    comet_logger = CometLogger(api_key="0", disabled=True, auto_metric_logging=False)

    # Comet logger configurations
    comet_logger.experiment.set_name(args.name)

    # Setup save dir
    if args.debug == 1:
        # A debug save_dir
        save_dir = 'experiments/test-debug'
        create_exp_dir(save_dir, visual_folder=True)
    else:
        # Create an experiment folder for logging and saving model's weights
        save_dir = 'experiments/{}-{}'.format(
            args.name,
            time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(save_dir, visual_folder=True)

    # Some args logic
    if args.debug == 1:
        print("Debug mode on.")
        args.fast_dev_run = True
        args.num_workers = 0

    # Set seed
    seed_everything(args.seed)

    # Get dataloader
    train_loader, valid_loader, test_loader = get_training_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        root_dir=args.dataset_dir, fold=args.fold)
    start_search_time = time.perf_counter()
    # ------ Supernet initialization ------
    with comet_logger.experiment.context_manager("search"):
        model = Supernet(encoder=args.nas_encoder, layers=args.nas_layers,
                         ops_set=args.nas_ops_set,
                         condition_metric=args.monitor_loss,
                         selection_epochs=args.nas_selection_epochs)

        # ------ Setup training -------
        # Setup checkpoint callbacks
        save_best_model = ModelCheckpoint(monitor=args.monitor_loss,
                                          dirpath=save_dir,
                                          filename='supernet_best_model', save_top_k=1,
                                          mode='max', save_last=False, verbose=True,
                                          save_weights_only=True)

        early_stop = EarlyStopping(
            monitor=args.monitor_loss,
            patience=50,
            mode='max',
            verbose=True
        )

        # Trainer
        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=save_dir,
                                                logger=comet_logger,
                                                callbacks=[save_best_model, early_stop])

        if len(args.nas_weight) == 0:
            # Train the supernet
            trainer.fit(model, train_loader, valid_loader)  # train

            # Save best model weights to comet
            comet_logger.experiment.log_asset(save_best_model.best_model_path)

            if not args.debug:
                model = model.load_from_checkpoint(save_best_model.best_model_path)
                trainer.validate(model, dataloaders=valid_loader)  # Just to re-attach the loaded model to trainer for saving weights later

        else:
            print("Skipping training the supernet. Using the specified weight file for the supernet.")
            model = model.load_from_checkpoint(args.nas_weight, strict=False)  # for compatibility with old codes
            trainer.validate(model, dataloaders=valid_loader)  # Just to re-attach the loaded model to trainer for saving weights later

        elapsed_time = time.perf_counter() - start_search_time
        print(f"The supernet training took {elapsed_time / 60 / 60:.4f} hours.")
        comet_logger.experiment.log_other('Train supernet time (hours)', elapsed_time / 60 / 60)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # Perform perturbation-based search
        model.find_best_arch(train_loader, valid_loader, device, args.nas_max_edges)

        trainer.save_checkpoint(save_dir+"/derived_model.ckpt", weights_only=True)
        comet_logger.experiment.log_asset(save_dir+"/derived_model.ckpt")
    elapsed_time = time.perf_counter() - start_search_time
    print(f"The search took {elapsed_time/60/60:.4f} hours.")
    comet_logger.experiment.log_other('Search time (hours)', elapsed_time/60/60)



    # ----- Train the candidate NAS now -------
    del model  # Remove the supernet
    with comet_logger.experiment.context_manager('derived'):
        print("Training the derived network now.")
        model = NASCandidate.load_from_checkpoint(save_dir+"/derived_model.ckpt", strict=False)

        # Plot the optimum DNN
        model.cleanup_optimum_dnn(remove_dead_node=True)
        fig = model.plot_arch()
        fig.show()
        comet_logger.experiment.log_figure(figure_name=f"Child DNN", figure=fig)

        # Save the number of trainable parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model trainable parameters: {params}")
        comet_logger.experiment.log_other('Derived model trainable parameters',
                                          params)

        # Setup checkpoint callbacks
        save_best_model = ModelCheckpoint(monitor=args.monitor_loss,
                                          dirpath=save_dir,
                                          filename='child_best_model', save_top_k=1,
                                          mode='max', save_last=False,
                                          verbose=True,
                                          save_weights_only=True)

        early_stop = EarlyStopping(
            monitor=args.monitor_loss,
            patience=100,
            mode='max',
            verbose=True
        )

        trainer = pl.Trainer.from_argparse_args(args, default_root_dir=save_dir,
                                                logger=comet_logger,
                                                callbacks=[save_best_model,
                                                           early_stop])

        # Train the supernet
        trainer.fit(model, train_loader, valid_loader)  # train

        # Save best model weights to comet
        comet_logger.experiment.log_asset(save_best_model.best_model_path)

        # Test the model
        if bool(args.debug):
            trainer.test(model=model, dataloaders=test_loader)  # test
        else:
            trainer.test(dataloaders=test_loader, ckpt_path='best')  # test

        # Check speed
        device = 'cuda' if args.gpus > 0 else 'cpu'
        speed = check_model_speed(model, device=device)
        comet_logger.experiment.log_metric('FPS', 1/speed)


if __name__ == '__main__':
    main()