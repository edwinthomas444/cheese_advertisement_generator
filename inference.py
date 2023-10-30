from transformers import GPT2Tokenizer
# from transformers import EncoderDecoderModel
from dataset.pipelines import GPTPipeline
from dataset.dataset import CheeseDescriptionsDataset, CheeseDescriptionsTemplateDataset
from torch.utils.data import DataLoader
from collators.collators import gpt_collator
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from models.gpt_neo import GPTNeoForCausalLM
from transformers import GPTNeoConfig
from tqdm import trange, tqdm
import torch
import random
import os


def main():
    print("Current device: ", torch.cuda.current_device())

    # innitialize tokenizer
    gpt_model_name = 'EleutherAI/gpt-neo-1.3B'

    tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name,
                                              bos_token='<|endoftext|>',
                                              eos_token='<|endoftext|>',
                                              pad_token='<|endoftext|>',
                                              sep_token='mask', # mask as sep
                                              padding_side = 'left')

    # innitialize dataset
    pipeline = GPTPipeline(tokenizer=tokenizer,
                           max_len_model=800,
                           max_len_context=450,
                           max_len_text=350)

    # data files load path
    annot_file_train = 'Data/slots_data/rhet_data_slots_cleaned_train.json'
    annot_file_valid = 'Data/slots_data/rhet_data_slots_cleaned_val.json'
    annot_file_test = 'Data/slots_data/rhet_data_slots_cleaned_test.json'
    template_file = 'configs/cheese_template.json'

    # load model from checkpoint
    model_path = '/home/edt000/u/cheese_advertisement_generator/checkpoints/model.bin'
    config_path = '/home/edt000/u/cheese_advertisement_generator/checkpoints/config_1.json'

    train_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_train,
                                              loader_pipeline=pipeline)

    valid_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_valid,
                                              loader_pipeline=pipeline)
    
    test_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_test,
                                              loader_pipeline=pipeline)
    
    inference_dataset = CheeseDescriptionsTemplateDataset(template_file=template_file,
                                                          loader_pipeline=pipeline)

    # innitalize loaders
    bs = 16
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        sampler=None,
        shuffle=True,  # enable shuffle of data
        collate_fn=gpt_collator,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=bs,
        sampler=None,
        shuffle=False,
        collate_fn=gpt_collator,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, # inference_dataset #test_dataset
        batch_size=bs,
        sampler=None,
        shuffle=False,
        collate_fn=gpt_collator,
        pin_memory=True
    )

    # params
    device = torch.device("cuda")

    
    gpt_cf = GPTNeoConfig.from_pretrained(config_path)
    model = GPTNeoForCausalLM.from_pretrained(model_path, config=gpt_cf)
    model.to(device)
    model.eval()

    #### TEST #####
    test_bar = tqdm(
        test_loader,
        desc='Iter (loss=X.XXX)',
        disable=False
    )
    
    with torch.no_grad():
        pred_list, inp_list, dec_list = [], [], []
        for val_step, val_batch in enumerate(test_bar):

            inp = [tens.to(device) for tens in val_batch]
            input_ids, attention_mask, labels, test_token_ids = inp

            # attention mask is based on current input length
            # as we are generating in autoregressive fashion from context
            # _update_model_kwargs_for_generation() in .generate() will auto append 1 to attention
            # mask for each newly generated token in the autoregressive generation
            # so only passing the innitial attention mask
            attention_mask = test_token_ids.new_ones(test_token_ids.shape)
            attention_mask[test_token_ids == tokenizer.pad_token_id] = 0
            # print('attention mask: ', attention_mask)

            preds = tokenizer.batch_decode(model.generate(
                    test_token_ids, 
                    attention_mask = attention_mask,
                    do_sample = True,
                    temperature = 0.5,
                    max_new_tokens = 250,
                    no_repeat_ngram_size = 3), skip_special_tokens=True)
            # ToDO: Explore constrained beam search decoding
            
            # preds = tokenizer.batch_decode(
                        #     model.generate(
                        #         test_token_ids,
                        #         do_sample=True,
                        #         temperature=0.9,
                        #         max_length=300), skip_special_tokens=False)
            
            # input = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
            dec_input = tokenizer.batch_decode(labels, skip_special_tokens = True)
            test_tokens = tokenizer.batch_decode(test_token_ids, skip_special_tokens = True)
            

            # pred
            pred_list.extend(preds)
            # prompt
            inp_list.extend(test_tokens)
            # gt
            dec_list.extend(dec_input)
    
    # save the predictions
    pred_save_dir = 'results/preds_gpt.txt'
    gt_save_dir = 'results/gt_gpt.txt'

    with open(pred_save_dir, 'w') as f, open(gt_save_dir, 'w') as f1:
        for inp, pred, gt in zip(inp_list, pred_list, dec_list):
            pred = pred.removeprefix(inp.strip())
            gt = gt.removeprefix(inp.strip())
            if not gt.strip():
                continue
            f.write(pred+"###")
            f1.write(gt+"###")
    
                    

if __name__ == '__main__':
    main()
