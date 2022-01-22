# -*- coding: utf-8 -*-
"""
run_finetune_qa
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BatteryBERT QA fine-tuning runner
author: Shu Huang (sh2009@cam.ac.uk)
"""
import os
import sys
import logging
import argparse
import transformers
from transformers import TrainingArguments, default_data_collator, EvalPrediction
import datasets
from datasets import load_metric
from batterybert.finetune import QAModel, FinetuneTokenizerFast, QADataset
from batterybert.finetune.utils import QuestionAnsweringTrainer, postprocess_qa_predictions

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse arguments from cli or defaults.

    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default='squad', type=str,
                        help="Default dataset: SQuAD v1.1.")
    parser.add_argument("--dataset_config_name", default=None, type=str,
                        help="Dataset config")
    parser.add_argument("--train_root", default=None, type=str,
                        help="Root of training dataset.")
    parser.add_argument("--eval_root", default=None, type=str,
                        help="Root of validation dataset.")

    # Required parameters.
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The pre-trained model name or path.")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output dir for checkpoints and logging.")

    parser.add_argument("--do_train", default=True, type=bool,
                        help="Training QA.")
    parser.add_argument("--do_eval", default=True, type=bool,
                        help="Evaluating QA.")

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


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=20,
        max_answer_length=30,
        null_score_diff_threshold=0.0,
        output_dir=args.output_dir,
        # log_level=log_level,
        prefix=stage,
    )

    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


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
    train_dataset = QADataset(args.model_name_or_path).get_train_dataset()
    eval_dataset = QADataset(args.model_name_or_path).get_eval_dataset()
    model = QAModel(args.model_name_or_path).get_model()
    data_collator = default_data_collator

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

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

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        eval_examples=eval_dataset if args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        #post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
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

    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        # logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": args.model_name_or_path, "tasks": "question-answering"}
    if args.dataset_name is not None:
        kwargs["dataset_tags"] = args.dataset_name
        if args.dataset_config_name is not None:
            kwargs["dataset_args"] = args.dataset_config_name
            kwargs["dataset"] = f"{args.dataset_name} {args.dataset_config_name}"
        else:
            kwargs["dataset"] = args.dataset_name

    return


if __name__ == '__main__':
    args = parse_arguments()

    if args.model_name_or_path is None:
        raise ValueError('--model_name_or_path must be provided via arguments or the '
                         'config file')
    if args.output_dir is None:
        raise ValueError('--output_dir must be provided via arguments or the '
                         'config file')

    main(args)
