"""
Author: Suresh Choudhary
Email: skcberlin@gmail.com
Date: 2024-10-14
Description: This script fine tunes the Bert Model using SQuAD dataset.
Version: 1.0
"""
import transformers
from datasets import load_dataset, DatasetDict
from datasets import ClassLabel, Sequence
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import mlflow
import torch
import collections
from tqdm.auto import tqdm
# Import metric loading from datasets
from evaluate import load
import numpy as np
import time
import psutil
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
#import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

def show_random_elements(dataset, num_examples=10):
    """
    Show a random sample of elements from the dataset.

    Args:
        dataset (Dataset): The dataset to sample from.
        num_examples (int, optional): The number of examples to sample. Default is 10.

    Raises:
        AssertionError: If num_examples is greater than the length of the dataset.
    """
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])

    # Apply transformation for ClassLabel and Sequence features
    for column, feature_type in dataset.features.items():
        if isinstance(feature_type, ClassLabel):
            df[column] = df[column].apply(lambda i: feature_type.names[i])
        elif isinstance(feature_type, Sequence) and isinstance(feature_type.feature, ClassLabel):
            df[column] = df[column].apply(lambda x: [feature_type.feature.names[i] for i in x])

    # Uncomment the following line to display the DataFrame as HTML if using in a notebook
    # display(HTML(df.to_html()))

    return df


def prepare_train_features(examples, tokenizer, pad_on_right, max_length, doc_stride):
    """
    Prepare tokenized training features for a dataset.

    Args:
        examples (dict): The dataset examples.
        tokenizer (Tokenizer): The tokenizer to use.
        pad_on_right (bool): Whether padding should occur on the right.
        max_length (int): Maximum length of the sequences.
        doc_stride (int): Stride length for overflowing tokens.

    Returns:
        dict: Tokenized examples with start and end positions for the answers.
    """
    # Strip left whitespace from questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize examples with truncation, padding, and handling overflows using a stride
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map each tokenized feature back to the original example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize lists for start and end positions
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    # Label the tokenized examples
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answer, set CLS token index as start and end
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start and end character index of the answer in the text
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find the token index corresponding to the start and end of the answer
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Check if answer is out of the span
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Adjust token indices to match answer boundaries
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def prepare_validation_features(examples, tokenizer, pad_on_right, max_length, doc_stride):
    """
    Prepares tokenized validation features for question-answering tasks.
    Removes extra whitespace from questions and tokenizes examples with overflow handling.

    Args:
        examples (dict): A dictionary containing the examples to tokenize.
        tokenizer (Tokenizer): Tokenizer object.
        pad_on_right (bool): Whether padding should occur on the right.
        max_length (int): Maximum length of tokenized inputs.
        doc_stride (int): Stride for splitting long contexts.

    Returns:
        dict: Tokenized examples with offset mappings and overflow handling.
    """
    # Remove left-side whitespace in questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize examples with truncation and padding
    tokenized_examples = tokenizer(
        text=examples["question"] if pad_on_right else examples["context"],
        text_pair=examples["context"] if pad_on_right else examples["question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Create a map from features to corresponding examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0
        sample_index = sample_mapping[i]

        # Store example id and adjust offset mapping
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def postprocess_qa_predictions(
    examples, tokenizer, features, raw_predictions, squad_v2, n_best_size=20, max_answer_length=30
):
    """
    Post-processes the raw predictions of a question-answering model to extract the best possible answer.

    Args:
        examples (dict): The original examples used for prediction.
        tokenizer (Tokenizer): Tokenizer object.
        features (list): Features for each example.
        raw_predictions (tuple): Model's raw start and end logits.
        squad_v2 (bool): Whether to apply SQuAD v2 specific processing.
        n_best_size (int): Number of best predictions to consider.
        max_answer_length (int): Maximum length for a valid answer.

    Returns:
        OrderedDict: Final predictions mapped by example id.
    """
    all_start_logits, all_end_logits = raw_predictions

    # Map example to corresponding features
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None  # Used only if squad_v2 is True
        valid_answers = []

        context = example["context"]

        # Iterate over all features for the current example
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]

            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue

                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        # Choose the best answer or null answer for SQuAD v2
        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions


def monitor_model(model):
    """
    Monitors model statistics such as parameter count and memory usage (GPU/CPU).

    Args:
        model (torch.nn.Module): Model to monitor.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory Allocated: {allocated:.2f} GB")
        print(f"GPU Memory Reserved: {reserved:.2f} GB")
    else:
        memory_info = psutil.virtual_memory()
        used_memory = memory_info.used / 1024**3
        total_memory = memory_info.total / 1024**3
        print(f"CPU Memory Usage: {used_memory:.2f} GB / {total_memory:.2f} GB")


def print_gpu_info():
    """
    Prints GPU information including memory usage for each available GPU.
    """
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 2:.2f} MB")
    else:
        print("No GPU available.")



def main():
    """Main function to fine-tune and evaluate a BERT model on the SQuAD dataset."""
    print_gpu_info()

    squad_v2 = False
    model_checkpoint = "bert-base-uncased"
    batch_size = 32
    n_best_size = 20

    # Load dataset from disk
    datasets = load_from_disk("./squad_data")

    # Preprocessing the training data
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    
    max_length = 384
    doc_stride = 128
    pad_on_right = tokenizer.padding_side == "right"

    # Tokenizing the datasets
    tokenized_datasets = datasets.map(
        lambda examples: prepare_train_features(
            examples, tokenizer, pad_on_right, max_length, doc_stride),
        batched=True,
        remove_columns=datasets["train"].column_names
    )

    # Model Fine-Tuning
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    model_name = model_checkpoint.split("/")[-1]

    # Start an MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_checkpoint", model_checkpoint)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("max_length", max_length)
        mlflow.log_param("num_train_epochs", 2)

        args = TrainingArguments(
            f"{model_name}-finetuned-squad-base",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=2,
            weight_decay=0.01,
        )

        data_collator = default_data_collator

        # Check for GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)  # Ensure the model is on the GPU

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Start timer
        start_time = time.time()

        # Run the training function
        trainer.train()

        # End timer
        elapsed_time = time.time() - start_time
        print(f"Time taken for fine-tuning: {elapsed_time:.2f} seconds")

        # Log the training time
        mlflow.log_metric("training_time", elapsed_time)

        # Evaluate the model
        validation_features = datasets["validation"].map(
            lambda examples: prepare_validation_features(
                examples, tokenizer, pad_on_right, max_length, doc_stride),
            batched=True,
            remove_columns=datasets["validation"].column_names
        )

        raw_predictions = trainer.predict(validation_features)
        validation_features.set_format(
            type=validation_features.format["type"], 
            columns=list(validation_features.features.keys())
        )

        max_answer_length = 30
        final_predictions = postprocess_qa_predictions(
            datasets["validation"], tokenizer, validation_features, raw_predictions.predictions, squad_v2
        )

        # Load the metric
        metric = load("squad")
        if squad_v2:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in final_predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v}
                for k, v in final_predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex["answers"]}
            for ex in datasets["validation"]
        ]
        result = metric.compute(predictions=formatted_predictions, references=references)

        # Log metrics
        for key, value in result.items():
            mlflow.log_metric(key, value)

        # Log the model
        mlflow.pytorch.log_model(model, "bert-base-model")

        print(result)


if __name__ == "__main__":
    main()





