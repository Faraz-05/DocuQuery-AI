
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_MODEL = "distilbert/distilbert-base-cased"
OUTPUT_DIR = Path("models/docuquery-qa")
SAMPLE_SIZE = 300
MAX_LENGTH = 384


def main():
    print("Loading a small SQuAD training sample...")

    dataset = load_dataset(
        "rajpurkar/squad",
        split=f"train[:{SAMPLE_SIZE}]",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def preprocess(examples):
        encoded = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            padding="max_length",
            return_offsets_mapping=True,
        )

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(encoded["offset_mapping"]):
            answers = examples["answers"][i]
            sequence_ids = encoded.sequence_ids(i)
            cls_index = encoded["input_ids"][i].index(
                tokenizer.cls_token_id
            )

            if not answers["answer_start"]:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            answer_start = answers["answer_start"][0]
            answer_end = answer_start + len(answers["text"][0])

            context_tokens = [
                j for j, sequence_id in enumerate(sequence_ids)
                if sequence_id == 1
            ]

            if not context_tokens:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            first_context = context_tokens[0]
            last_context = context_tokens[-1]

            # If truncation removed the answer, use the CLS position.
            if (
                offsets[first_context][0] > answer_start
                or offsets[last_context][1] < answer_end
            ):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            token_start = first_context
            while (
                token_start <= last_context
                and offsets[token_start][0] <= answer_start
            ):
                token_start += 1

            token_end = last_context
            while (
                token_end >= first_context
                and offsets[token_end][1] >= answer_end
            ):
                token_end -= 1

            start_positions.append(token_start - 1)
            end_positions.append(token_end + 1)

        encoded.pop("offset_mapping")

        encoded["start_positions"] = start_positions
        encoded["end_positions"] = end_positions

        return encoded

    print("Tokenizing questions, contexts, and answer spans...")

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )

    print("Loading the pretrained DistilBERT model...")

    model = AutoModelForQuestionAnswering.from_pretrained(BASE_MODEL)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving the fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("Training complete!")
    print(f"Model saved at: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
