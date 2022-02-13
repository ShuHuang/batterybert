# -*- coding: utf-8 -*-
"""
run_pretrain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BatteryBERT pretrain runner
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import sys
import logging
import argparse
import datasets
import transformers
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from batterybert.pretrain import PretrainTokenizer, PretrainModel, PretrainDataset

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments from cli or defaults.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()

    # Optional json config to override defaults below
    parser.add_argument("--checkpoint", default='bert-base-cased', type=str,
                        help="The BatteryBERT checkpoint containing the config file")

    # Required parameters.
    parser.add_argument("--train_root", default=None, type=str,
                        help="The input data dir of training text file.")
    parser.add_argument("--eval_root", default=None, type=str,
                        help="The input data dir of evaluation text file.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")
    parser.add_argument("--tokenizer_root", default=None, type=str,
                        help="The tokenizer vocab dir.")

    # Masking Parameters
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help='Probability of masked tokens per sequence')

    # Training Configuration
    parser.add_argument(
        "--num_steps_per_checkpoint",
        type=int,
        default=10000,
        help="Number of update steps between writing checkpoints.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--fp16", default=True, action='store_true',
                        help="Use PyTorch AMP training")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False,
                        help="Overwrite output directory")
    parser.add_argument("--no_cuda", type=bool, default=False,
                        help="Use CPU or GPU")


    # Hyperparameters
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--per_device_train_batch_size", default=32, type=int,
                        help="Per-device batch size for training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="The weight decay value.")
    parser.add_argument("--save_steps", default=10000, type=float,
                        help="Number of saved steps.")
    parser.add_argument("--save_total_limits", default=1000000, type=float,
                        help="The maximum limits of saved steps.")
    parser.add_argument("--num_train_epochs", default=40, type=int,
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

    return args


def main(args):
    """
    Run pretraining
    :param args: parsed arguments
    :return:
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    tokenizer = PretrainTokenizer(args.tokenizer_root).get_tokenizer()
    lm_datasets = PretrainDataset(args.train_root, args.eval_root, args.tokenizer_root).get_tokenized_datasets()
    model = PretrainModel(args.checkpoint).get_model()
    model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        prediction_loss_only=args.prediction_loss_only,
        no_cuda=args.no_cuda,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    trainer.train()
    trainer.save_model(args.output_dir)

    return


if __name__ == '__main__':
    args = parse_arguments()

    if args.train_root is None:
        raise ValueError('--train_root must be provided via arguments or the '
                         'config file')
    if args.eval_root is None:
        raise ValueError('--eval_root must be provided via arguments or the '
                         'config file')
    if args.output_dir is None:
        raise ValueError('--output_dir must be provided via arguments or the '
                         'config file')

    main(args)
