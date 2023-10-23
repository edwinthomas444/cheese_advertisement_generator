from transformers import BertTokenizer
# from transformers import EncoderDecoderModel
from dataset.pipelines import BertPipeline
from dataset.dataset import CheeseDescriptionsDataset
from torch.utils.data import DataLoader
from collators.collators import bert_base_collator
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from models.encoder_decoder import EncoderDecoderModel
from tqdm import trange, tqdm
import torch
import random


def main():
    # innitialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', lowercase=True)

    # innitialize dataset
    pipeline = BertPipeline(tokenizer=tokenizer,
                            max_len_decoder=512,
                            max_len_encoder=512)

    annot_file_train = 'Data/slots_data/rhet_data_slots_cleaned_train.json'
    annot_file_valid = 'Data/slots_data/rhet_data_slots_cleaned_val.json'
    annot_file_test = 'Data/slots_data/rhet_data_slots_cleaned_test.json'

    train_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_train,
                                              loader_pipeline=pipeline)

    valid_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_valid,
                                              loader_pipeline=pipeline)
    
    test_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_test,
                                              loader_pipeline=pipeline)

    # innitalize loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        sampler=None,
        shuffle=True,  # enable shuffle of data
        collate_fn=bert_base_collator,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=16,
        sampler=None,
        shuffle=False,
        collate_fn=bert_base_collator,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        sampler=None,
        shuffle=False,
        collate_fn=bert_base_collator,
        pin_memory=True
    )

    # params
    learning_rate = 1e-05
    gradient_accumulation_steps = 1
    warmup_proportion = 0.1
    epochs = 100
    validate_steps = 10
    device = torch.device("cuda", 0)

    # innitialize model from pretrained checkpoint
    encoder = 'bert-base-uncased'
    decoder = 'bert-base-uncased'
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder, decoder)
    # decoder configs (greedy search innitialization)
    model.config.min_length = 1
    model.config.max_length = 130
    model.config.eos_token_id = [tokenizer.sep_token_id, tokenizer.pad_token_id]
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.num_beams = 1
    model.config.num_return_sequences = 1
    model.config.vocab_size = tokenizer.vocab_size
    model.to(device)

    # innitialize optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        correct_bias=False
    )

    # linnitialize learning rate scheduler
    total_steps = int(len(train_loader)*epochs/gradient_accumulation_steps)
    num_warmup_steps = int(total_steps*warmup_proportion)
    print(
        f'\nTotal Steps: {total_steps} | Total Warmup Steps: {num_warmup_steps}')
    

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    # start training cycle
    for ep in trange(0,
                     epochs,
                     desc="Epoch",
                     disable=False):
        
        print('\n Epoch: ', ep)
        #### TRAIN #####
        train_bar = tqdm(
            train_loader,
            desc='Iter (loss=X.XXX)',
            disable=False
        )
        model.train()
        for train_step, batch in enumerate(train_bar):
            inp = [tens.to(device) for tens in batch]
            input_ids, attention_mask, cross_attention_mask, decoder_input_ids, decoder_attention_mask, labels = inp

            # obtain loss
            model_out = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                            #   cross_attention_mask=cross_attention_mask,
                              decoder_input_ids=decoder_input_ids,
                              decoder_attention_mask=decoder_attention_mask,
                              labels=labels)

            # ignore mlm and phloss
            loss = model_out.loss

            # update bar
            train_bar.set_description(
                'Iter (loss=%5.3f)' % loss.item()
            )
            # model update after every step (batch)
            loss.backward()
            # use scheduler updated lr
            # calling optimizer.step() before scheduler.step() following PyTorch 1.1 recomm.
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


            # validation
            if (train_step+1)%validate_steps == 0:
                #### VALID #####
                valid_bar = tqdm(
                    valid_loader,
                    desc='Iter (loss=X.XXX)',
                    disable=False
                )
                model.eval()
                
                with torch.no_grad():
                    val_loss_total = 0.0
                    pred_list, inp_list, dec_list = [], [], []
                    for val_step, val_batch in enumerate(valid_bar):
                        if val_step != 0:
                            continue
                        inp = [tens.to(device) for tens in val_batch]
                        input_ids, attention_mask, cross_attention_mask, decoder_input_ids, decoder_attention_mask, labels = inp
                        # obtain loss
                        model_out = model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        # cross_attention_mask=cross_attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_attention_mask,
                                        labels=labels)

                        valid_loss = model_out.loss
                        val_loss_total+= valid_loss
                        valid_bar.set_description(
                            'Iter (loss=%5.3f)' % valid_loss.item()
                        )
                        # get the generated text in list of size valid batch size
                        preds = tokenizer.batch_decode(model.generate(
                            input_ids), skip_special_tokens=True)
                        input = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
                        dec_input = tokenizer.batch_decode(decoder_input_ids, skip_special_tokens = True)

                        pred_list.extend(preds)
                        inp_list.extend(input)
                        dec_list.extend(dec_input)
                        

                    # random display
                    # disp_ind = random.randint(0, len(inp_list)-1)
                    disp_ind = 0
                    display_inp, display_pred, display_gt = inp_list[disp_ind], pred_list[disp_ind], dec_list[disp_ind]
                    val_loss_total = val_loss_total/len(valid_loader)
                    print('\n\n Input Text: ', display_inp)
                    print('\n Pred: ', display_pred)
                    print('\n Actual: ', display_gt)
                    print(f'\n Total val loss: {val_loss_total}')





if __name__ == '__main__':
    main()
