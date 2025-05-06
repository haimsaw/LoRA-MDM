import torch.nn as nn
import os

def load_bert(model_path, extend_tokens=False):
    bert = BERT(model_path, extend_tokens)
    bert.eval()
    bert.text_model.training = False
    for p in bert.parameters():
        p.requires_grad = False
    return bert

class BERT(nn.Module):
    def __init__(self, modelpath: str, extend_tokens=False):
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        
        if extend_tokens:
            print("Extending tokenizer")
            self.tokenizer.add_tokens(['sks', 'hta', 'oue', 'asar', 'nips'])
            self.text_model = AutoModel.from_pretrained('save/extended_bert/')
                                                        
        # Text model
        else:
            self.text_model = AutoModel.from_pretrained(modelpath)
            
    def forward(self, texts):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
        mask = encoded_inputs.attention_mask.to(dtype=bool)
        # output = output * mask.unsqueeze(-1)
        return output, mask
