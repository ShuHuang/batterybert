# -*- coding: utf-8 -*-
"""
run_finetune_doc_classify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BatteryBERT sequence classification fine-tuning runner
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import sys
import logging
import argparse
import datasets
import transformers
from transformers import Trainer, TrainingArguments, default_data_collator
from sklearn.metrics import accuracy_score
from batterybert.finetune import DocClassModel, FinetuneTokenizerFast, PaperDataset

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments from cli or defaults.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()


    # Required parameters.
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The pre-trained model name or path.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")
    parser.add_argument("--train_root", default=None, type=str,
                        help="Root of training dataset.")
    parser.add_argument("--eval_root", default=None, type=str,
                        help="Root of validation dataset.")

    # Training Configuration
    parser.add_argument(
        "--num_steps_per_checkpoint",
        type=int,
        default=200,
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
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--per_device_train_batch_size", default=12, type=int,
                        help="Per-device batch size for training.")
    parser.add_argument("--per_device_eval_batch_size", default=12, type=int,
                        help="Per-device batch size for validation.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="The weight decay value.")
    parser.add_argument("--save_steps", default=10000, type=float,
                        help="Number of saved steps.")
    parser.add_argument("--save_total_limits", default=20000, type=float,
                        help="The maximum limits of saved steps.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Number of training epochs")
    parser.add_argument("--prediction_loss_only", default=True, type=bool,
                        help="Prediction loss or whole loss")
    parser.add_argument("--evaluation_strategy", default="steps", type=str,
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


def main(args):
    """
    Run pretraining
    :param args: parsed arguments
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    tokenizer = FinetuneTokenizerFast(args.model_name_or_path).get_tokenizer()
    train_dataset, eval_dataset = PaperDataset(args.model_name_or_path, args.train_root, args.eval_root).get_dataset()
    model = DocClassModel(args.model_name_or_path).get_model()
    data_collator = default_data_collator

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
        weight_decay=args.weight_decay
    )

    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # the callback that computes metrics of interest
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    trainer.train()

    trainer.evaluate()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    return


if __name__ == '__main__':
    args = parse_arguments()

    if args.model_name_or_path is None:
        raise ValueError('--model_name_or_path must be provided via arguments or the '
                         'config file')
    if args.output_dir is None:
        raise ValueError('--output_dir must be provided via arguments or the '
                         'config file')
    if args.train_root is None:
        raise ValueError('--train_root must be provided via arguments or the '
                         'config file')
    if args.eval_root is None:
        raise ValueError('--eval_root must be provided via arguments or the '
                         'config file')

    main(args)
