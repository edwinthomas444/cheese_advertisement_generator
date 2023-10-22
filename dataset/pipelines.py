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
        encoder_attention_mask = [1 if x!=0 else 0 for x in encoder_input_ids]
        encoder_cross_attention_mask = [1 if x!=0 else 0 for x in encoder_attention_mask]

        # prepare decoder inputs
        dec_tokens = self.tokenizer(decoder_text,
                                    max_length = self.max_len_decoder,
                                    padding = 'max_length',
                                    truncation = True)
        
        decoder_input_ids = dec_tokens['input_ids']
        decoder_attention_mask = [1 if x!=0 else 0 for x in decoder_input_ids]

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


        
    

