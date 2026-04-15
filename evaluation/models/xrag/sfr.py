import torch
from torch import Tensor
from transformers import MistralModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class SFR(MistralModel):
    def get_embed_dim(self):
        return self.config.hidden_size
    
    def get_embed_length(self):
        return 1
    
    def get_embedding(self,input_ids,attention_mask):
        outputs = self.forward(input_ids=input_ids,attention_mask=attention_mask)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        return embeddings
    
    def get_doc_embedding(self,input_ids,attention_mask):
        return self.get_embedding(input_ids,attention_mask)
    
    def get_query_embedding(self,input_ids,attention_mask):
        return self.get_embedding(input_ids,attention_mask)
