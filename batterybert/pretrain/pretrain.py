# -*- coding: utf-8 -*-
"""
batterybert.pretrain.pretrain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BatteryBERT pretrain runner
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import json
import argparse


def parse_arguments():
    """Parse arguments from cli, json file or defaults.
    Parsing precedent: cli_args > config_file > argparse defaults

    :return: parsed arguments
    :rtype: parser.parse_args()
    """
    parser = argparse.ArgumentParser()

    # Optional json config to override defaults below
    parser.add_argument("--config_file", default=None, type=str,
                        help="JSON config for overriding defaults")

    # Required parameters.
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="The input data dir of text file.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")
    parser.add_argument("--model_config_file", default=None, type=str,
                        help="The BatteryBERT model config")

    # Masking Parameters
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help='Probability of masked tokens per sequence')

    # Training Configuration
    parser.add_argument(
        '--num_steps_per_checkpoint',
        type=int,
        default=200,
        help="Number of update steps between writing checkpoints.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Use PyTorch AMP training")

    # Hyperparameters
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--per_device_train_batch_size", default=32, type=int,
                        help="Per-device batch size for training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="The weight decay value.")
    parser.add_argument("--save_steps", default=10000, type=float,
                        help="Number of saved steps.")
    parser.add_argument("--save_total_limits", default=20000, type=float,
                        help="The maximum limits of saved steps.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Number of training epochs")
    parser.add_argument("--prediction_loss_only", default=True, type=bool,
                        help="Prediction loss or whole loss")
    parser.add_argument("--evaluation_strategy", default="epoch", type=str,
                        help="Evaluation strategy")

    # Set by torch.distributed.launch
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    args = parser.parse_args()
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    # Distinguish arguments that were found in sys.argv[1:]
    aux_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    for arg in vars(args):
        aux_parser.add_argument('--' + arg)
    cli_args, _ = aux_parser.parse_known_args()

    # Config file arguments
    if args.config_file is not None:
        with open(args.config_file) as jf:
            configs = json.load(jf)
        for key in configs:
            if key in vars(args) and key not in vars(cli_args):
                setattr(args, key, configs[key])
    return args
