#Methods for preprocessing and fine-tuning DistilBERT for text classification
from collections import defaultdict
from datasets import Dataset

from transformers import DistilBertTokenizerFast
from datasets import Dataset,  concatenate_datasets
import pandas as pd
from tqdm import tqdm
import numpy as np


# processing data into chunks of 512 tokens
def preprocess_with_labels(df, max_length=512, chunk_size=1000):
    """
    Tokenize a large pandas DataFrame in batches and split documents into chunks.
    Creates a Dataset with only input_ids, attention_mask, and labels.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    all_rows = []

    for start in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
        end = min(start + chunk_size, len(df))
        texts = df.iloc[start:end]["fullText"].tolist()

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,   
            return_attention_mask=True
        )

        for input_ids in tokenized["input_ids"]:
            # split into 512-token chunks
            chunks = [input_ids[j:j+max_length] for j in range(0, len(input_ids), max_length)]
            masks = [[1]*len(chunk) for chunk in chunks]

            for chunk, mask in zip(chunks, masks):
                all_rows.append({
                    "input_ids": chunk,
                    "attention_mask": mask,
                    "labels": chunk.copy()
                })

    # Build HF dataset directly from dicts
    chunked_df = pd.DataFrame(all_rows)
    dataset = Dataset.from_pandas(chunked_df, preserve_index=False)

    # ensure format is tensor-friendly
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


#eralier version of the function above
def preprocess_funktion_fast_old(df, chunk_size=1000, max_length=512):
    """
    Tokenize and chunk a large pandas DataFrame safely in smaller batches.
    Returns a Hugging Face Dataset object.
    """
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    
    datasets_list = []  # store smaller datasets
    
    # Process dataframe in batches
    for start in tqdm(range(0, len(df), chunk_size), desc="Processing chunks"):
        end = min(start + chunk_size, len(df))
        sub_df = df.iloc[start:end].copy()
        
        # Tokenize with fast tokenizer
        tokenized = tokenizer(
            sub_df["fullText"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_attention_mask=True
        )
        
        # Handle overflowing tokens
        input_ids = tokenized["input_ids"]
        attention_masks = tokenized["attention_mask"]
        sample_mapping = tokenized["overflow_to_sample_mapping"]
        
        chunked_rows = []
        for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
            original_idx = sample_mapping[i]
            row = sub_df.iloc[original_idx].to_dict()
            row.update({
                "chunk_id": i,
                "input_ids": ids,
                "attention_mask": mask
            })
            chunked_rows.append(row)
        
        chunked_df = pd.DataFrame(chunked_rows)
        
        # Convert small chunked_df to Dataset and append
        datasets_list.append(Dataset.from_pandas(chunked_df))
    
    # Concatenate all smaller datasets into one
    final_dataset = concatenate_datasets(datasets_list)
    return final_dataset


# older version of the function above, indexes data chunks
from transformers import DistilBertTokenizerFast
def preprocess_funktion_fast_older(df):
    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    tokenized = tokenizer(
        df["fullText"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
        return_attention_mask=True
    )
    # tokenized has extra metadata: "overflow_to_sample_mapping"
    input_ids = tokenized["input_ids"]
    attention_masks = tokenized["attention_mask"]
    sample_mapping = tokenized["overflow_to_sample_mapping"]

    chunked_rows = []
    for i, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        original_idx = sample_mapping[i]
        row = df.iloc[original_idx].to_dict()  # copy all metadata
        row.update({
            "chunk_id": i,          # unique chunk index
            "input_ids": ids,       # the 512-token chunk
            "attention_mask": mask  # mask
        })
        chunked_rows.append(row)

    chunked_df = pd.DataFrame(chunked_rows)
    return chunked_df

from transformers import DistilBertTokenizer
from transformers import DistilBertTokenizerFast

# earliest version of the function above, slow and creates column of overflowing tokens which are basically not truncated
def preprocess_function(examples):
    return tokenizer(examples['fullText'], 
                    padding="max_length", 
                    truncation=True, 
                    stride=128, 
                    return_overflowing_tokens=True,
                    return_special_tokens_mask=True)

# Apply preprocessing


# first 512 tokein in column token id, rest is in overflowing tokenid
# Group chunks back to documents
def group_chunks(dataset):
    grouped = defaultdict(lambda: {"fullText": None, "input_ids": [], "attention_mask": []})
    
    for row in dataset:
        doc_id = row["id"]  # or whatever identifies one document
        if grouped[doc_id]["fullText"] is None:
            grouped[doc_id]["fullText"] = row["fullText"]
        grouped[doc_id]["input_ids"].append(row["input_ids"])
        grouped[doc_id]["attention_mask"].append(row["attention_mask"])
    
    # convert back to HF dataset
    grouped_list = [
        {"id": doc_id, 
        "fullText": v["fullText"], 
        "input_ids": v["input_ids"], 
        "attention_mask": v["attention_mask"]}
        for doc_id, v in grouped.items()
    ]
    return Dataset.from_list(grouped_list)
