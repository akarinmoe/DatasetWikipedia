from datasets import load_dataset
import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

ds = load_dataset("wikimedia/wikipedia", "20231101.en")

ds.save_to_disk("./datasets")