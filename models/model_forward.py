
# stores utilities for forward pass through model
# specific to model forward() funtion
from dataclasses import dataclass
import torch


@dataclass
class ForwardPassOutput:
    loss: torch.FloatTensor = None
    input_ids: torch.LongTensor = None
    attention_mask: torch.LongTensor = None
    labels: torch.LongTensor = None
    test_token_ids: torch.LongTensor = None
    
# Bert model
def bert_base_encdec_forward_pass(model, inp):
    input_ids, attention_mask, cross_attention_mask, decoder_input_ids, decoder_attention_mask, labels = inp
    # obtain loss
    model_out = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      # cross_attention_mask=cross_attention_mask,
                      decoder_input_ids=decoder_input_ids,
                      decoder_attention_mask=decoder_attention_mask,
                      labels=labels)
    loss = model_out.loss

    output = ForwardPassOutput(
        loss = loss,
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = labels
    )
    return output

# GPT-NEO
def gpt_neo_dec_forward_pass(model, inp):
    input_ids, attention_mask, labels, test_token_ids = inp

    # obtain loss
    model_out = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      labels=labels)

    loss = model_out.loss

    output = ForwardPassOutput(
        loss = loss,
        input_ids = input_ids,
        attention_mask = attention_mask,
        test_token_ids = test_token_ids,
        labels = labels
    )
    return output


# Bert model
def T5_forward_pass(model, inp):
    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels = inp
    # obtain loss
    model_out = model(input_ids=input_ids,
                      attention_mask=attention_mask,
                      decoder_input_ids=decoder_input_ids,
                      decoder_attention_mask=decoder_attention_mask,
                      labels=labels)
    loss = model_out.loss

    output = ForwardPassOutput(
        loss = loss,
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = decoder_input_ids # as labels has -100 for <pad>, .generate() cant decode it, so we pass decoder_input_ids
    )
    return output