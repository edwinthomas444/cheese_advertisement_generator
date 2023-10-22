from transformers import BertTokenizer
from transformers import EncoderDecoderModel
from dataset.pipelines import BertPipeline
from dataset.dataset import CheeseDescriptionsDataset
from torch.utils.data import DataLoader
from collators.collators import bert_base_collator
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from tqdm import trange, tqdm
import torch


def main():
    # innitialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', lowercase=True)

    # innitialize dataset
    pipeline = BertPipeline(tokenizer=tokenizer,
                            max_len_decoder=512,
                            max_len_encoder=512)

    annot_file = 'Data/slots_data/rhet_data_slots_cleaned.json'
    train_dataset = CheeseDescriptionsDataset(annotation_file=annot_file,
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


    # params
    learning_rate = 1e-05
    gradient_accumulation_steps = 1
    warmup_proportion = 0.01
    epochs = 5
    device = torch.device("cuda", 0)

    # innitialize model from pretrained checkpoint
    encoder = 'bert-base-uncased'
    decoder = 'bert-base-uncased'
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder, decoder)
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


if __name__ == '__main__':
    main()
