from models.encoder_decoder import EncoderDecoderModel
from models.gpt_neo import GPTNeoForCausalLM
from transformers import EncoderDecoderConfig, GPTNeoConfig


def bert_base_loader(config, model_type, tokenizer, train=True):
    if train:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            config['model'][model_type]['name'],
            config['model'][model_type]['name']
        )
        model.config.vocab_size = tokenizer.vocab_size
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.sep_token_id = tokenizer.sep_token_id
    else:
        # load from pretrained checkpoint
        config_path = config['checkpoints']['config_path']
        model_path = config['checkpoints']['bin_path']
        enc_dec_cf = EncoderDecoderConfig.from_pretrained(config_path)
        model = EncoderDecoderModel.from_pretrained(
            model_path, config=enc_dec_cf)
        
    return model


def gpt_neo_loader(config, model_type, tokenizer, train=True):
    if train:
        model = GPTNeoForCausalLM.from_pretrained(
            config['model'][model_type]['name']
        )
        model.config.vocab_size = tokenizer.vocab_size
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.sep_token_id = tokenizer.sep_token_id
    else:
        # load from pretrained checkpoint
        config_path = config['checkpoints']['config_path']
        model_path = config['checkpoints']['bin_path']
        gpt_cf = GPTNeoConfig.from_pretrained(config_path)
        model = GPTNeoForCausalLM.from_pretrained(
            model_path,
            config = gpt_cf
        )

    return model
