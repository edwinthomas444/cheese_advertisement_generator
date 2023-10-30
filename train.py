
from dataset.dataset import CheeseDescriptionsDataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from tqdm import trange, tqdm
import torch
import random
import json
import argparse
from models.model_configs import *
import os
# saving checkpoints


def save_checkpoint(
        save_dir,
        model,
        epoch
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    output_model_file = os.path.join(save_dir, f"model.bin")
    # saving epoch meta data with config file name
    output_config_file = os.path.join(save_dir, f"config.json")

    model.eval()
    model_save = model.module if hasattr(model, 'module') else model
    config_save = model.module.config if hasattr(
        model, 'module') else model.config
    # save model checkpoint
    torch.save(model_save.state_dict(), output_model_file)
    # save config
    config_save.to_json_file(output_config_file)


def driver(args, config):

    # innitialize tokenizer
    model_type = config['type']
    config_obj = eval(model_type)

    tokenizer = config_obj['tokenizer'].from_pretrained(
        config['model'][model_type]['name'],
        **config['model'][model_type]['tokenizer']
    )

    # innitialize dataset
    pipeline = config_obj['data_pipeline'](tokenizer=tokenizer,
                                           len_output=config['model'][model_type
                                                                      ]['max_len_output'],
                                           len_context=config['model'][model_type]['max_len_context'])

    annot_file_train = config['data']['train']
    annot_file_valid = config['data']['valid']
    annot_file_test = config['data']['test']

    train_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_train,
                                              loader_pipeline=pipeline)

    valid_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_valid,
                                              loader_pipeline=pipeline)

    test_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_test,
                                             loader_pipeline=pipeline)

    bs = config['train']['batch_size']

    # innitalize loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        sampler=None,
        shuffle=True,  # enable shuffle of data
        collate_fn=config_obj['collator'],
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=bs,
        sampler=None,
        shuffle=False,
        collate_fn=config_obj['collator'],
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=bs,
        sampler=None,
        shuffle=False,
        collate_fn=config_obj['collator'],
        pin_memory=True
    )

    # load model
    device = torch.device("cuda", 0)
    model = config_obj['model_loader'](
        config=config,
        model_type=model_type,
        tokenizer=tokenizer,
        train=True
    )
    model = torch.nn.DataParallel(model)
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
        lr=config['train']['learning_rate'],
        correct_bias=False
    )

    # linnitialize learning rate scheduler
    epochs = config['train']['epochs']
    total_steps = int(len(train_loader)*epochs /
                      config['train']['gradient_accumulation_steps'])
    num_warmup_steps = int(total_steps*config['train']['warmup'])
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

        model_forward_fnc = config_obj['model_forward']
        model.train()
        for train_step, batch in enumerate(train_bar):
            inp = [tens.to(device) for tens in batch]

            model_forward = model_forward_fnc(model=model,
                                              inp=inp)

            loss = model_forward.loss.mean()

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
            if (train_step+1) % config['valid']['validate_steps'] == 0:
                #### VALID #####
                valid_bar = tqdm(
                    valid_loader,
                    desc='Iter (loss=X.XXX)',
                    disable=False
                )
                model.eval()
                val_loss_min = 1e08
                with torch.no_grad():
                    val_loss_list = []
                    pred_list, inp_list, dec_list = [], [], []
                    for val_step, val_batch in enumerate(valid_bar):
                        inp = [tens.to(device) for tens in val_batch]
                        model_forward = model_forward_fnc(model=model,
                                                          inp=inp)

                        valid_loss = model_forward.loss.mean()

                        val_loss_list.append(valid_loss)
                        valid_bar.set_description(
                            'Iter (loss=%5.3f)' % valid_loss.item()
                        )

                        # prepare inputids and attention mask differently for "encoder-decoder" and "decoder-only" models
                        if model.module.config.is_encoder_decoder:
                            # Encoder Decoder model
                            generate_input_ids = model_forward.input_ids
                            attention_mask = model_forward.attention_mask
                        else:
                            # Decoder Only models
                            generate_input_ids = model_forward.test_token_ids
                            attention_mask = generate_input_ids.new_ones(
                                generate_input_ids.shape)
                            attention_mask[generate_input_ids ==
                                           tokenizer.pad_token_id] = 0

                        # generate with generation params passed through the config
                        preds = tokenizer.batch_decode(
                            model.module.generate(
                                generate_input_ids,
                                attention_mask=attention_mask,
                                **config['generation_configs'][args.generation]), skip_special_tokens=False)

                        input = tokenizer.batch_decode(
                            model_forward.input_ids, skip_special_tokens=True)
                        dec_input = tokenizer.batch_decode(
                            model_forward.labels, skip_special_tokens=True)

                        pred_list.extend(preds)
                        inp_list.extend(input)
                        dec_list.extend(dec_input)

                    # random display
                    disp_ind = random.randint(0, len(inp_list)-1)
                    print('\n Display ind: ', disp_ind)
                    # disp_ind = 0
                    display_inp, display_pred, display_gt = inp_list[
                        disp_ind], pred_list[disp_ind], dec_list[disp_ind]
                    val_loss_total = sum(val_loss_list)/len(val_loss_list)

                    print('\n\n Input Text: ', display_inp)
                    print('\n Pred: ', display_pred)
                    print('\n Actual: ', display_gt)
                    print(f'\n Total val loss: {val_loss_total}')

                    # saving checkpoints
                    if val_loss_total <= val_loss_min:
                        val_loss_min = val_loss_total
                        # save the checkpoint
                        print(
                            f'\n Saving Checkpoint for val loss: {val_loss_min}')
                        save_checkpoint(save_dir=config['train']['save_dir'],
                                        model=model,
                                        epoch=ep)


def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    # argument parser stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--generation', type=str,
                        required=True,
                        help='Generate type (matched with config generate options eg: greedy)')

    args = parser.parse_args()
    config = config_parser(args.config)

    # invoke the generic training driver
    driver(args, config)


if __name__ == '__main__':
    main()
