"""Convert multihop JSONL to HuggingFace DatasetDict format (test only)."""
import json
from datasets import Dataset, DatasetDict, Features, Value, Sequence

INPUT_JSONL = "data/art_e_vince_kaminski_multihop/hf_dataset.jsonl"
OUTPUT_DIR = "data/art_e_vince_kaminski_multihop"

CUSTODIAN_MAP = {"kaminski-v": "vince.kaminski@enron.com"}

# Read multihop JSONL
records = []
with open(INPUT_JSONL) as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        evidence = row.get("evidence_mails", [])
        message_ids = [f"<{m}>" for m in evidence]

        custodian = row.get("custodian_id", "kaminski-v")
        inbox_address = CUSTODIAN_MAP.get(custodian, f"{custodian}@enron.com")

        records.append({
            "id": i,
            "question": row["question"],
            "answer": row["answer"],
            "message_ids": message_ids,
            "how_realistic": None,
            "inbox_address": inbox_address,
            "query_date": "",
        })

print(f"Total records: {len(records)}")
print(f"Sample record: {records[0]}")

# Define features matching existing dataset
features = Features({
    "id": Value("int32"),
    "question": Value("string"),
    "answer": Value("string"),
    "message_ids": Sequence(Value("string")),
    "how_realistic": Value("float32"),
    "inbox_address": Value("string"),
    "query_date": Value("string"),
})

# Create test-only DatasetDict
test_ds = Dataset.from_list(records, features=features)
ds_dict = DatasetDict({"test": test_ds})

# Save
ds_dict.save_to_disk(OUTPUT_DIR)
print(f"\nSaved to {OUTPUT_DIR}")

# Verify
loaded = DatasetDict.load_from_disk(OUTPUT_DIR)
print(f"Loaded splits: {list(loaded.keys())}")
test = loaded["test"]
print(f"Test size: {len(test)}")
print(f"Features: {test.features}")
print(f"Sample: {test[0]}")
