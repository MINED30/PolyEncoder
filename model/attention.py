import torch
import torch.nn as nn
import torch.nn.functional as F

# 그냥 dot attention
def dot_attention(q, k, v, v_mask=None, dropout=None):
  attention_weights = torch.matmul(q, k.transpose(-1, -2))
  if v_mask is not None:
    extended_v_mask = (1.0 - v_mask.unsqueeze(1)) * -100000.0
    attention_weights += extended_v_mask
  attention_weights = F.softmax(attention_weights, -1)
  if dropout is not None:
    attention_weights = dropout(attention_weights)
  output = torch.matmul(attention_weights, v)
  return output