# methods for the embedding calc pipeline
import pandas as pd
import json
import nltk
from nltk.corpus import stopwords
import torch
import torch.nn.functional as F


# method for splitting input text into chunks of 512 tokens
def chunk_text(tokenizer,text, max_len=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i+max_len-2] for i in range(0, len(tokens), max_len-2)]
    return [tokenizer.build_inputs_with_special_tokens(chunk) for chunk in chunks]




# method for splitting input text into chunks of 512 tokens bi-transformer model
def chunk_text_bi(tokenizer, text, max_len=512):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    alphanumeric_tokens = [
        token for token in tokens 
        if tokenizer.decode([token]).strip().isalnum()
    ]
    chunks = [alphanumeric_tokens[i:i+max_len-2] for i in range(0, len(alphanumeric_tokens), max_len-2)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# takes a list of text chunks, tokenizes them, and returns a DataFrame with token embeddings
def embTokenDf_bi(chunks, tokenizer, model):
    df_outputs = pd.DataFrame()
    i = 0
    for chunk in chunks:
        # tokenizer to get tensors for the model
        if i % 1000 == 0:
            print(f"Processing chunk {i+1}/{len(chunks)}")
        encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        # shape [1, seq_len, hidden_size]
        embeddings = output.last_hidden_state.squeeze(0)  # -> [seq_len, hidden_size]

        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))

        # add tokens to df
        df_chunk = pd.DataFrame(embeddings.detach().numpy())
        df_chunk.insert(0, "token", tokens)

        df_outputs = pd.concat([df_outputs, df_chunk], ignore_index=True)
        i+= 1
    print("Processed chunks:", i)

    return df_outputs


# take a list of text chunks and returns a df with token embeddings
def embTokenDf_distil(chunks, tokenizer, model):
    df_outputs = pd.DataFrame()

    for chunk in chunks:
        encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        # encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        # like [batch_size, seq_len, hidden_dim]
        embeddings = output.last_hidden_state.squeeze(0)  # shape: [seq_len, 768]
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0))

        df_chunk = pd.DataFrame(embeddings.cpu().numpy())
        df_chunk.insert(0, "token", tokens)

        df_outputs = pd.concat([df_outputs, df_chunk], ignore_index=True)
        print(f"Processed {len(tokens)} tokens")

    return df_outputs

#bekanntes problem dass die fulltext spalte meherere einzele strings erhält statt einen ganzen
#mit neuen datasource nicht nötig
def join_strings(list):
    text = ''
    for _ in list:
        text += _
    return text

def embTokenDf(chunks, model, tokenizer):
    df_outputs = pd.DataFrame()
    for chunk in chunks:
        inputs_ids = torch.tensor([chunk])
        attention_mask = torch.ones_like(inputs_ids)
        inputs = {"input_ids": inputs_ids, "attention_mask": attention_mask}

        with torch.no_grad():
            output = model(**inputs)

        embeddings = output.last_hidden_state.squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(chunk)

        df_chunk = pd.DataFrame(embeddings.detach().numpy())
        df_chunk.insert(0, "token", tokens)
        df_outputs = pd.concat([df_outputs, df_chunk], ignore_index=True)
        print("Processed embeddings:", len(embeddings))
        print("Processed tokens: ", len(tokens))
    return df_outputs

# calc avg embdding to get embedding archetype for whole year
def average_token_embeddings(df):
    non_embedding_cols = ['year', 'token']
    embedding_cols = [c for c in df.columns if c not in non_embedding_cols]
    
    df_avg = (
        df.groupby('token').agg(year=('year', 'first'),
              **{col: 'mean' for col in embedding_cols}).reset_index()
    )
    
    df_avg = df_avg[['year', 'token'] + embedding_cols]
    return df_avg


#check for similar tokens in two df
def check_similars(text1, text2):
    similars = []
    for token_a in text1['token']:
        for token_b in text2['token']:
            if token_a == token_b and token_b not in similars:
                similars.append(token_a)
    return similars

# compute dinstances between two df
def compute_distances(df1, df2, similars):
    distances = []
    for token in similars:
        if token in df1['token'].values and token in df2['token'].values:
            emb1 = torch.tensor(df1[df1['token'] == token].iloc[:, 1:].values.flatten(), dtype=torch.float)
            emb2 = torch.tensor(df2[df2['token'] == token].iloc[:, 1:].values.flatten(), dtype=torch.float)
            cos_sim = F.cosine_similarity(emb1, emb2, dim=0)
            distances.append((token, cos_sim))
    return pd.DataFrame(distances, columns=['token', 'distance'])

#same but for single df
def compute_distances_single_df(df1):
    distances = []
    emb1 = torch.tensor(df1.iloc[0, 1:].astype(float).values.flatten(), dtype=torch.float)
    for index, row in df1.iterrows():
        print(f"Processing row {index+1}{row['token']}")
        emb2 = torch.tensor(row[1:].astype(float).values.flatten(), dtype=torch.float)

        cos_sim = F.cosine_similarity(emb1, emb2, dim=0)
        distances.append((row['token'], cos_sim.item()))
    return pd.DataFrame(distances, columns=['token', 'distance'])