from datasets import load_from_disk, concatenate_datasets
from prompts.binary_classification import preprocess_input
from preprocessing.entries import Entries

dataset_path_harmful = "data/raw/jbb-behaviors/harmful"
dataset_path_benign = "data/raw/jbb-behaviors/benign"

dataset_harmful = load_from_disk(dataset_path_harmful)
dataset_benign = load_from_disk(dataset_path_benign)

output_path = "data/processed/benchmark/jbb-behaviors"

print("dataset_harmful[0]:", dataset_harmful[0])
print("dataset_benign[0]:", dataset_benign[0])

def preprocess_harmful(example):
    return {
        Entries.ORIGINAL_CONTENT: example["Goal"],
        Entries.BINARY_CLASSIFICATION_PROMPT: preprocess_input(example["Goal"]),
        Entries.BINARY_EXPECTED_BOXED_ANSWER: "A",
    }

def preprocess_benign(example):
    return {
        Entries.ORIGINAL_CONTENT: example["Goal"],
        Entries.BINARY_CLASSIFICATION_PROMPT: preprocess_input(example["Goal"]),
        Entries.BINARY_EXPECTED_BOXED_ANSWER: "B",
    }

dataset_harmful = dataset_harmful.map(preprocess_harmful, remove_columns="Goal")
dataset_benign = dataset_benign.map(preprocess_benign, remove_columns="Goal")
processed_dataset = concatenate_datasets([dataset_harmful, dataset_benign])

processed_dataset.save_to_disk(output_path)
print("Processed Dataset shapes:", len(processed_dataset))

