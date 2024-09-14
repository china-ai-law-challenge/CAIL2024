import os
import argparse
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import seed_everything
from .dataset import LJPDataModule
from .model import LJPBertModule

def main(args):
    # for reproducibility
    seed_everything(args.seed, workers=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # use your local path, or download from huggingface with "bert-base-chinese"
    tokenizer_path = "./models/bert-base-chinese"
    logger = TensorBoardLogger(args.log_path, name="bert-base-chinese")
    if args.do_train:
        # setup datamodule
        mdljp_data_module_train = LJPDataModule(
            tokenizer_path=tokenizer_path,
            train_dataset_path=args.train_data_path,
            val_dataset_path=args.val_data_path,
            test_dataset_path=args.test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        mdljp_data_module_train.setup(stage="fit")
        # setup model
        # use your local path, or download from huggingface with "bert-base-chinese"
        mdljp_predictor = LJPBertModule("./models/bert-base-chinese", lr=args.lr)
        trainer = pl.Trainer(
            devices=[0,1],
            accelerator="gpu",
            max_epochs=args.epochs,
            logger=logger,
            val_check_interval=1.0,  # 1.0, not 1
            callbacks=[EarlyStopping(monitor="final_score", mode="max")],
            accumulate_grad_batches=args.accumulate_grad_batches,
            deterministic=True,  # for reproducibility
        )
        trainer.fit(mdljp_predictor, datamodule=mdljp_data_module_train)
        # (optional) save final ckpt
        trainer.save_checkpoint(f"./outputs/bert-base-chinese/epoch_{args.epochs}_lr_{args.lr}_bs_{args.batch_size}.ckpt")
        
        if args.do_test:
            # test directly after train, lightning will load the best model automatically
            mdljp_data_module_train.setup(stage="test")
            trainer.test(dataloaders=mdljp_data_module_train.test_dataloader())
    elif args.do_test:
        mdljp_data_module_test = LJPDataModule(
            tokenizer_path=tokenizer_path,
            test_dataset_path=args.test_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        mdljp_data_module_test.setup(stage="test")
        mdljp_predictor = LJPBertModule.load_from_checkpoint(
            args.model_load_path,
            map_location=lambda storage, loc: storage.cuda(1),
        )
        trainer = pl.Trainer(devices=[1], num_nodes=1, max_epochs=1, logger=logger, deterministic=True)
        trainer.test(mdljp_predictor, datamodule=mdljp_data_module_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./outputs/lightning_logs/",
        help="log path",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="train data path",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default=None,
        help="val data path",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
        help="test data path",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
    )
    parser.add_argument(
        "--model_load_path",
        type=str,
        help="model load path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="gradient accumulation batches",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="epochs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=30,
        help="num workers for dataloaders",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate"
    )
    args = parser.parse_args()
    main(args)