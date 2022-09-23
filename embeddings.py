import torch
from transformers import BertTokenizer, BertModel
from config import *
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def get_embeddings(df, device):
  words = df
  
  marked_texts = []
  
  for i in words:
    marked_text = "[CLS] " + i + " [SEP]"
    marked_texts.append(marked_text)

  tokenized = []
  for i in marked_texts:
    tokenized_text = tokenizer.tokenize(i)
    tokenized.append(tokenized_text)

  index_token = []

  for i in tokenized:
    index_token.append(tokenizer.convert_tokens_to_ids(i))
  
  segments = []

  for i in tokenized:
    segments.append([1] * len(i))

  tokens_tensors = []
  for i in index_token:
    tokens_tensor = torch.tensor([i])
    tokens_tensors.append(tokens_tensor)

  segment_tensors = []
  for i in segments:
    segments_tensors = torch.tensor([i])
    segment_tensors.append(segments_tensors)

  model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, 
                                  ).to(device)

  model.eval()

  output = []
  hidden_state = []
  for i in range(len(tokens_tensors)):

    with torch.no_grad():

      outputs = model(tokens_tensors[i], segment_tensors[i])
      output.append(outputs)

      hidden_states = outputs[2]
      hidden_state.append(hidden_states)

  embeddings = []
  for i in range(len(hidden_state)):
    token_vecs = hidden_state[i][-2][0]
    embedding = torch.mean(token_vecs, dim=0)
    embeddings.append(embedding)

  return embeddings