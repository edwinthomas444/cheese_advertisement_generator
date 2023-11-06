from models.encoder_decoder import EncoderDecoderModel
from models.gpt_neo import GPTNeoForCausalLM
from models.t5 import T5ForConditionalGeneration
from transformers import EncoderDecoderConfig, GPTNeoConfig, T5Config


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
        model.config.seq2set = False
    else:
        # load from pretrained checkpoint
        config_path = config['checkpoint']['config_path']
        model_path = config['checkpoint']['bin_path']
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
        model.config.seq2set = False
    else:
        # load from pretrained checkpoint
        config_path = config['checkpoint']['config_path']
        model_path = config['checkpoint']['bin_path']
        gpt_cf = GPTNeoConfig.from_pretrained(config_path)
        model = GPTNeoForCausalLM.from_pretrained(
            model_path,
            config = gpt_cf
        )

    return model


def t5_base_loader(config, model_type, tokenizer, train=True):
    if train:
        model = T5ForConditionalGeneration.from_pretrained(
            config['model'][model_type]['name']
        )
        model.config.vocab_size = tokenizer.vocab_size
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.seq2set = False
    else:
        # load from pretrained checkpoint
        config_path = config['checkpoint']['config_path']
        model_path = config['checkpoint']['bin_path']
        enc_dec_cf = T5Config.from_pretrained(config_path)
        model = T5ForConditionalGeneration.from_pretrained(
            model_path, config=enc_dec_cf)
        
    return model