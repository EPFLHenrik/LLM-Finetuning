from datasets import load_from_disk
from prompts.binary_classification import preprocess_input
from preprocessing.entries import Entries

dataset_path = "data/raw/advbench/train"
dataset = load_from_disk(dataset_path)

output_path = "data/processed/benchmark/advbench"

print("dataset[0]:", dataset[0])

def preprocess_example(example):
    original_content = example["content"][0]
    return {
        Entries.ORIGINAL_CONTENT: original_content,
        Entries.BINARY_CLASSIFICATION_PROMPT: preprocess_input(original_content),
        Entries.BINARY_EXPECTED_BOXED_ANSWER: "A",
    }

# For advbench, all the expected boxed answers are "A" (Harmful)
processed_dataset = dataset.map(preprocess_example, remove_columns="content")

print("Processed Dataset shapes:", len(processed_dataset))
print("Processed Dataset[0]:", processed_dataset[0])

processed_dataset.save_to_disk(output_path)
