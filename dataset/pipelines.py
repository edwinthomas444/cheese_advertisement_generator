class BertPipeline:
    def __init__(self,
                 tokenizer,
                 len_context,
                 len_output,
                 **kwargs):
        self.tokenizer = tokenizer
        self.max_len_encoder = len_context
        self.max_len_decoder = len_output

    def __call__(self, row):
        decoder_text = row['output']
        encoder_text = row['input']

        # prepare encoder inputs
        enc_tokens = self.tokenizer(encoder_text,
                                    max_length = self.max_len_encoder,
                                    padding = 'max_length',
                                    truncation = True)
        encoder_input_ids = enc_tokens['input_ids']
        encoder_attention_mask = [1 if x!=self.tokenizer.pad_token_id else 0 for x in encoder_input_ids]
        encoder_cross_attention_mask = [1 if x!=self.tokenizer.pad_token_id else 0 for x in encoder_attention_mask]

        # prepare decoder inputs
        dec_tokens = self.tokenizer(decoder_text,
                                    max_length = self.max_len_decoder,
                                    padding = 'max_length',
                                    truncation = True)
        
        # automatically adds [CLS] at beginning and ends by [PAD] token
        decoder_input_ids = dec_tokens['input_ids']
        # print('\n decoder input ids: ', decoder_input_ids)
        decoder_attention_mask = [1 if x!=self.tokenizer.pad_token_id else 0 for x in decoder_input_ids]

        # prepare the labels and target ids are shifted inside the decoder model forward pass
        decoder_target_ids = [x for x in decoder_input_ids]
        
        ds = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'cross_attention_mask': encoder_cross_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': decoder_target_ids
        }

        return ds

class GPTPipeline:
    def __init__(self,
                 tokenizer,
                 len_context,
                 len_output,
                 **kwargs):
        self.tokenizer = tokenizer
        self.max_len_context = len_context
        self.max_len_text = len_output
        self.max_len_model = self.max_len_context + self.max_len_text

    def __call__(self, row):
        context, text = row['input'], row['output']

        context = self.tokenizer.tokenize(context)[:self.max_len_context-2]
        text = self.tokenizer.tokenize(text)[:self.max_len_text-1]

        # [self.tokenizer.sep_token]
        combined_tokens = [self.tokenizer.bos_token] + context + [self.tokenizer.sep_token] + text + [self.tokenizer.eos_token]
        
        test_tokens = [self.tokenizer.bos_token] + context + [self.tokenizer.sep_token]
        
        # combined_tokens = [self.tokenizer.bos_token] + context + text + [self.tokenizer.eos_token]
        
        # test_tokens = [self.tokenizer.bos_token] + context
        
        # truncate and pad tokens
        combined_tokens = combined_tokens[:self.max_len_model]
        pad_token_len = self.max_len_model - len(combined_tokens)
        # apply left-padding instead
        combined_tokens = [self.tokenizer.pad_token]*(pad_token_len) + combined_tokens

        token_ids = self.tokenizer.convert_tokens_to_ids(combined_tokens)
        test_token_ids = self.tokenizer.convert_tokens_to_ids(test_tokens)

        # prepare the labels and target ids are shifted inside the decoder model forward pass
        decoder_target_ids = [x for x in token_ids]
        
        attention_mask = [1 if x!=self.tokenizer.pad_token_id else 0 for x in token_ids]

        ds = {
            'input_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': decoder_target_ids,
            'test_token_ids': test_token_ids
        }

        return ds


# T5 pipeline
class T5Pipeline:
    def __init__(self,
                 tokenizer,
                 len_context,
                 len_output,
                 **kwargs):
        self.tokenizer = tokenizer
        self.max_len_encoder = len_context
        self.max_len_decoder = len_output

    def __call__(self, row):
        decoder_text = row['output']
        encoder_text = row['input']

        # prepare encoder inputs
        # no sos token and ends with eos token ('</s>')
        enc_tokens = self.tokenizer(encoder_text,
                                    max_length = self.max_len_encoder,
                                    padding = 'max_length',
                                    truncation = True)
        encoder_input_ids = enc_tokens['input_ids']
        encoder_attention_mask = [1 if x!=self.tokenizer.pad_token_id else 0 for x in encoder_input_ids]

        # prepare decoder inputs
        # add pad token in front as .generate() uses it in the beginning of the text _prepare_decoder_input_ids_for_generation()
        # from the model.config.decoder_start_token_id which is same as pad or 0 for this model
        dec_tokens = self.tokenizer(decoder_text,
                                    max_length = self.max_len_decoder-1,
                                    padding = 'max_length',
                                    truncation = True)
        
        decoder_input_ids = [self.tokenizer.pad_token_id] + dec_tokens['input_ids']

        # replace pad tokens by -100 and add sos token as Pad or 0
        decoder_target_ids = [-100 if (x==0 and i!=0) else x for i, x in enumerate(decoder_input_ids)]
        # print('decoder intput ids: ', decoder_input_ids)
        decoder_attention_mask = [1 if x!=-100 else 0 for x in decoder_target_ids]
        
        ds = {
            'input_ids': encoder_input_ids,
            'attention_mask': encoder_attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': decoder_attention_mask,
            'labels': decoder_target_ids
        }

        return ds
    

