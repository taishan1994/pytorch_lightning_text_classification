import os
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")
import pytorch_lightning as pl
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from model import ModelInterface
from data import DataInterface
from utils import load_model_path_by_args


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks


def main(args):
    pl.seed_everything(args.seed)
    load_path = load_model_path_by_args(args)
    print("loading model path:{} ...".format(load_path))
    data_module = DataInterface(**vars(args))

    if load_path is None:
        model = ModelInterface(**vars(args))
    else:
        model = ModelInterface(**vars(args))
        args.resume_from_checkpoint = load_path

    # # If you want to change the logger's saving folder
    # logger = TensorBoardLogger(save_dir='kfold_log', name=args.log_dir)
    # args.callbacks = load_callbacks()
    # args.logger = logger

    trainer = Trainer.from_argparse_args(args)
    if args.do_train:
        """
        LightningDeprecationWarning: Setting `Trainer(resume_from_checkpoint=)` 
        is deprecated in v1.5 and will be removed in v1.7. 
        Please pass `Trainer.fit(ckpt_path=)` directly instead
        """
        trainer.fit(model, data_module)
    if args.do_test:
        print('开始测试')
        trainer.test(model, data_module,
                     ckpt_path="./lightning_logs/version_35/checkpoints/epoch=6-step=39374.ckpt")

    if args.do_predict:
        print('开始预测')
        predictions = trainer.predict(model, data_module,
                                      ckpt_path="./lightning_logs/version_35/checkpoints/epoch=6-step=39374.ckpt")
        # 这里只打印一个batch的
        y_pred, y_true = predictions[0]
        print(y_pred)
        print(y_true)

if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    # Training Info
    parser.add_argument('--dataset', default='simple_data', type=str)
    parser.add_argument('--data_dir', default='data/ref/cnews', type=str)
    parser.add_argument('--model_name', default='simple_model', type=str)
    parser.add_argument('--loss', default='ce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    # Model Hyperparameters
    parser.add_argument('--vocab_size', default=0, type=int)
    parser.add_argument('--input_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--output_dim', default=10, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--biFlag', default=True, type=bool)
    parser.add_argument('--dropout', default=0.1, type=int)

    # Other
    # 这里添加其它的参数
    parser.add_argument('--class_num', default=10, type=int)

    # Add pytorch lightning's args to parser as a group.
    parser = Trainer.add_argparse_args(parser)

    ## Deprecated, old version
    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=100)

    args = parser.parse_args()
    with open(os.path.join(args.data_dir, 'vocab.txt'), 'r', encoding='utf-8') as fp:
        vocab = fp.read().strip().split('\n')
    vocab = vocab
    token2id = {}
    id2token = {}
    for k, v in enumerate(vocab):
        token2id[v] = k
        id2token[k] = v
    args.token2id = token2id
    args.id2token = id2token
    args.vocab_size = len(vocab)

    args.do_train = False
    args.do_test = True
    args.do_predict = True
    if args.do_test or args.do_predict:
        args.load_v_num = '35'
    # pprint(vars(args))
    main(args)
