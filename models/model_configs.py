# Define model_type to config mappings 
# to be matched with configs/base.json

from models.model_forward import bert_base_encdec_forward_pass, gpt_neo_dec_forward_pass, T5_forward_pass
from collators.collators import bert_base_collator, gpt_collator, T5_collator
from dataset.pipelines import BertPipeline, GPTPipeline, T5Pipeline
from transformers import BertTokenizer, GPT2Tokenizer, T5Tokenizer
from loaders.base_loaders import bert_base_loader, gpt_neo_loader, t5_base_loader

# bert transformer configs
bert_base_transformer = {
    "model_forward": bert_base_encdec_forward_pass,
    "collator": bert_base_collator,
    "data_pipeline": BertPipeline,
    "tokenizer": BertTokenizer,
    "model_loader": bert_base_loader
}


# GPT-Neo config
gpt_neo = {
    "model_forward": gpt_neo_dec_forward_pass,
    "collator": gpt_collator,
    "data_pipeline": GPTPipeline,
    "tokenizer": GPT2Tokenizer,
    "model_loader": gpt_neo_loader
}

# T5 config
t5_base = {
    "model_forward": T5_forward_pass,
    "collator": T5_collator,
    "data_pipeline": T5Pipeline,
    "tokenizer": T5Tokenizer,
    "model_loader": t5_base_loader
}
