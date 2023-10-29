class BertPipeline:
    def __init__(self,
                 tokenizer,
                 max_len_encoder,
                 max_len_decoder,
                 **kwargs):
        self.tokenizer = tokenizer
        self.max_len_encoder = max_len_encoder
        self.max_len_decoder = max_len_decoder

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
                 max_len_context,
                 max_len_text,
                 max_len_model,
                 **kwargs):
        self.tokenizer = tokenizer
        self.max_len_context = max_len_context
        self.max_len_text = max_len_text
        self.max_len_model = max_len_model

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


        
    


        
    

