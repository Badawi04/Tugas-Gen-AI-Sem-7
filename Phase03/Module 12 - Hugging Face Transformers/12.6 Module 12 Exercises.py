# 12.6 Module 12 Exercises (Halaman 143)
"""Benchmark pipeline, token budget, NER, dan fine-tuning LoRA."""

import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)


class ModelBenchmark:
    """Menjalankan prompt yang sama pada dua pipeline text-generation."""

    def __init__(self, model_names: tuple[str, str], **pipeline_kwargs: Any):
        self.model_names = model_names
        self.pipelines = {
            name: pipeline(
                "text-generation",
                model=name,
                tokenizer=name,
                **pipeline_kwargs,
            )
            for name in model_names
        }

    def run(self, prompts: list[str], max_new_tokens: int = 40) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for prompt in prompts:
            for model_name, generator in self.pipelines.items():
                started = time.perf_counter()
                output = generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                )[0]["generated_text"]
                latency_ms = (time.perf_counter() - started) * 1000
                rows.append(
                    {
                        "model": model_name,
                        "prompt": prompt,
                        "latency_ms": round(latency_ms, 2),
                        "output_length": len(output.split()),
                        "output": output,
                    }
                )
        return rows

    @staticmethod
    def print_table(rows: list[dict[str, Any]]) -> None:
        print(f"{'Model':30} {'Latency(ms)':>12} {'Words':>8}  Prompt")
        print("-" * 85)
        for row in rows:
            print(
                f"{row['model'][:30]:30} {row['latency_ms']:12.2f} "
                f"{row['output_length']:8}  {row['prompt']}"
            )


class TokenBudgetChecker:
    def __init__(self, model_name: str = "gpt2", max_tokens: int = 100):
        if max_tokens < 0:
            raise ValueError("max_tokens harus >= 0")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def filter_texts(self, texts: list[str]) -> list[tuple[str, int]]:
        remaining: list[tuple[str, int]] = []
        for text in texts:
            token_count = len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
            if token_count <= self.max_tokens:
                remaining.append((text, token_count))
        return remaining


def extract_entities_from_corpus(
    texts: list[str],
    model_name: str = "dslim/bert-base-NER",
) -> dict[str, list[str]]:
    """Mengumpulkan entity PERSON, ORG, dan LOC yang unik dari seluruh teks."""
    ner = pipeline(
        "ner",
        model=model_name,
        tokenizer=model_name,
        aggregation_strategy="simple",
    )
    entities: dict[str, set[str]] = {"PERSON": set(), "ORG": set(), "LOC": set()}
    for text in texts:
        for entity in ner(text):
            entity_type = str(entity["entity_group"]).upper()
            if entity_type in entities:
                entities[entity_type].add(str(entity["word"]).strip())
    return {entity_type: sorted(values) for entity_type, values in entities.items()}


QA_PAIRS = [
    ("What is RAG?", "RAG means Retrieval-Augmented Generation."),
    ("What is fine-tuning?", "Fine-tuning adapts a pretrained model with task data."),
    ("What is an embedding?", "An embedding is a vector representation of text."),
    ("What is a vector database?", "A vector database stores embeddings for similarity search."),
    ("What is LangChain?", "LangChain is a framework for building LLM applications."),
    ("What is LlamaIndex?", "LlamaIndex connects language models to external data."),
    ("What is prompt engineering?", "Prompt engineering designs instructions for better outputs."),
    ("What is a tokenizer?", "A tokenizer converts text into model tokens."),
    ("What is a transformer?", "A transformer is a neural architecture based on attention."),
    ("What is LoRA?", "LoRA fine-tunes models using low-rank adapter matrices."),
    ("What is PEFT?", "PEFT fine-tunes only a small number of model parameters."),
    ("What is NER?", "NER identifies named entities in text."),
    ("What is a language model?", "A language model predicts tokens from context."),
    ("What is inference?", "Inference is using a trained model to produce predictions."),
    ("What is supervised learning?", "Supervised learning uses labelled examples."),
    ("What is a dataset?", "A dataset is a structured collection of examples."),
    ("What is an epoch?", "An epoch is one complete pass through training data."),
    ("What is a batch?", "A batch is a group of examples processed together."),
    ("What is model latency?", "Latency is the time required to produce a response."),
    ("What is zero-shot learning?", "Zero-shot learning performs a task without examples."),
]


def fine_tune_lora(
    output_dir: str | Path = "lora_qna_output",
    model_name: str = "distilgpt2",
    epochs: int = 3,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Melatih adapter LoRA pada 20 pasangan Q&A dan menyimpannya."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            target_modules=["c_attn"],
            lora_dropout=0.1,
            bias="none",
        ),
    )

    examples = [
        {"question": question, "answer": answer, "text": f"Question: {question} Answer: {answer}"}
        for question, answer in QA_PAIRS
    ]
    dataset = Dataset.from_list(examples)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=[])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return peft_model, tokenizer, examples


def verify_fine_tuned_model(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    max_new_tokens: int = 24,
) -> list[dict[str, Any]]:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
    )
    results = []
    for example in examples:
        prompt = f"Question: {example['question']} Answer:"
        answer = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )[0]["generated_text"].strip()
        expected_terms = set(example["answer"].lower().split()[:3])
        correct = bool(expected_terms.intersection(answer.lower().split()))
        results.append({"question": example["question"], "answer": answer, "correct": correct})
    return results


if __name__ == "__main__":
    prompts = [
        "Explain retrieval augmented generation:",
        "Define an embedding:",
        "What does a tokenizer do?",
        "Explain model inference:",
        "What is LoRA used for?",
        "Define named entity recognition:",
        "What is a vector database?",
        "Explain supervised learning:",
        "What is a transformer?",
        "Define prompt engineering:",
    ]
    benchmark = ModelBenchmark(("distilgpt2", "gpt2"))
    benchmark.print_table(benchmark.run(prompts))

    checker = TokenBudgetChecker(max_tokens=20)
    print("\nTexts within budget:", checker.filter_texts(prompts))

    corpus = [
        "Alice works at OpenAI in San Francisco.",
        "Bob visited Microsoft headquarters in Seattle.",
    ]
    print("\nEntities:", extract_entities_from_corpus(corpus))

    lora_model, lora_tokenizer, examples = fine_tune_lora()
    verification = verify_fine_tuned_model(lora_model, lora_tokenizer, examples)
    print("\nLoRA verification:")
    for result in verification:
        print(result)
