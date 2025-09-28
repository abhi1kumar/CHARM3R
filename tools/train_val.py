"""
    Sample Run:
    CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/gup.yaml --resume_model=output/gup/checkpoints/checkpoint_epoch_140.pth --ext height30 --eval_adapt --shift_coord_conv 1. 0 0. 0. 1. -98.03 0 0 1. -e
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse
import torch
import numpy as np
import random

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from datetime import datetime

parser = argparse.ArgumentParser(description='implementation of DEVIANT')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--config', type=str, default = 'experiments/config.yaml')
parser.add_argument('--resume_model', type=str, default=None)
parser.add_argument('--ext', type=str, default=None)
parser.add_argument('--eval_adapt', action= 'store_true', default= False, help= 'eval_adapt')
parser.add_argument('--shift_coord_conv', nargs= 9, metavar=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'), type= float, default= None, help= 'shift coordinate convolution')
parser.add_argument('--depth_pp', type=float, default=0.0, help='depth pre-processing')
args = parser.parse_args()

def create_logger(log_file):
    # Remove all handlers associated with the root logger object.
    # See https://stackoverflow.com/a/49202811
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console)
    return logging.getLogger(__name__)

def init_torch(rng_seed, cuda_seed):
    """
    Initializes the seeds for ALL potential randomness, including torch, numpy, and random packages.

    Args:
        rng_seed (int): the shared random seed to use for numpy and random
        cuda_seed (int): the random seed to use for pytorch's torch.cuda.manual_seed_all function
    """
    # seed everything
    os.environ['PYTHONHASHSEED'] = str(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    torch.cuda.manual_seed(cuda_seed)
    torch.cuda.manual_seed_all(cuda_seed)

    # make the code deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pretty_print(name, input, val_width=40, key_width=0):
    """
    This function creates a formatted string from a given dictionary input.
    It may not support all data types, but can probably be extended.

    Args:
        name (str): name of the variable root
        input (dict): dictionary to print
        val_width (int): the width of the right hand side values
        key_width (int): the minimum key width, (always auto-defaults to the longest key!)

    Example:
        pretty_str = pretty_print('conf', conf.__dict__)
        pretty_str = pretty_print('conf', {'key1': 'example', 'key2': [1,2,3,4,5], 'key3': np.random.rand(4,4)})

        print(pretty_str)
        or
        logging.info(pretty_str)
    """
    pretty_str = name + ': {\n'
    for key in input.keys(): key_width = max(key_width, len(str(key)) + 4)

    for key in input.keys():
        val = input[key]
        # round values to 3 decimals..
        if type(val) == np.ndarray: val = np.round(val, 3).tolist()
        # difficult formatting
        val_str = str(val)
        if len(val_str) > val_width:
            # val_str = pprint.pformat(val, width=val_width, compact=True)
            val_str = val_str.replace('\n', '\n{tab}')
            tab = ('{0:' + str(4 + key_width) + '}').format('')
            val_str = val_str.replace('{tab}', tab)
        # more difficult formatting
        format_str = '{0:' + str(4) + '}{1:' + str(key_width) + '} {2:' + str(val_width) + '}\n'
        pretty_str += format_str.format('', key + ':', val_str)

    # close root object
    pretty_str += '}'
    return pretty_str

def main():  
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    exp_parent_dir = os.path.join(cfg['trainer']['log_dir'], os.path.basename(args.config).split(".")[0])
    cfg['trainer']['log_dir'] = exp_parent_dir
    logger_dir     = os.path.join(exp_parent_dir, "log")
    os.makedirs(exp_parent_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = create_logger(os.path.join(logger_dir, timestamp))

    if args.ext is not None:
        cfg['dataset']['ext_config'] = args.ext
    if args.eval_adapt:
        if 'eval_adapt' in cfg['model'].keys():
            cfg['model']['eval_adapt'] = True
            if args.shift_coord_conv is None:
                cfg['model']['shift_coord_conv'] = np.eye(3).tolist()
            else:
                cfg['model']['shift_coord_conv'] = np.array(args.shift_coord_conv).reshape((3,3)).tolist()
            cfg['model']['depth_pp'] = args.depth_pp
            print(cfg['model']['shift_coord_conv'])

    pretty = pretty_print('conf', cfg)
    logging.info(pretty)
    # init torch
    init_torch(rng_seed= cfg['random_seed']-3, cuda_seed= cfg['random_seed'])
    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])
    if cfg['dataset']['type'] != 'carla':
        test_loader = None

    # build model
    model = build_model(cfg,train_loader.dataset.cls_mean_size)

    # evaluation mode
    if args.evaluate:
        if args.resume_model is None:
            raise FileNotFoundError
        else:
            cfg['tester']['resume_model'] = args.resume_model
        tester = Tester(cfg, model, val_loader, logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      test_loader2= test_loader)
    trainer.train()
    tester = Tester(cfg, model, val_loader, logger)
    tester.test()


if __name__ == '__main__':
    main()
