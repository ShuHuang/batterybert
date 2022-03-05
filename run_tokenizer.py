# -*- coding: utf-8 -*-
"""
run_pretrain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BatteryBERT pretrain runner
author: Shu Huang (sh2009@cam.ac.uk)
"""
import sys
import logging
import argparse
from pathlib import Path
from tokenizers import BertWordPieceTokenizer


logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments from cli or defaults.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_root", default=None, type=str,
                        help="Root of training text.")
    parser.add_argument("--save_root", default=None, type=str,
                        help="Root of saving directory.")
    parser.add_argument("--save_name", default=None, type=str,
                        help="Name of trained tokenizer.")

    # Wordpiece parameters
    parser.add_argument("--lowercase", default=True, type=bool,
                        help="Do lower case or upper case.")
    parser.add_argument("--vocab_size", default=30522, type=int,
                        help="Vocabulary size.")
    parser.add_argument("--min_frequency", default=2, type=int,
                        help="Minimum frequency.")
    parser.add_argument("--limit_alphabet", default=1000, type=int,
                        help="Number of limited alphabet")
    parser.add_argument("--wordpieces_prefix", default="##", type=str,
                        help="wordpieces prefix")
    parser.add_argument("--special_tokens", default=['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], type=list,
                        help="Special tokens")

    args = parser.parse_args()

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
    """
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    paths = [str(x) for x in Path(args.train_root).glob('**/*.txt')]

    # initialize
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=args.lowercase
    )

    # train
    tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=args.min_frequency,
                    limit_alphabet=args.limit_alphabet, wordpieces_prefix=args.wordpieces_prefix,
                    special_tokens=args.special_tokens)

    tokenizer.save_model(args.save_root, args.save_name)


if __name__ == '__main__':
    args = parse_arguments()

    if args.train_root is None:
        raise ValueError('--train_root must be provided via arguments or the '
                         'config file')
    if args.save_root is None:
        raise ValueError('--save_root must be provided via arguments or the '
                         'config file')
    if args.save_name is None:
        raise ValueError('--save_name must be provided via arguments or the '
                         'config file')

    main(args)
