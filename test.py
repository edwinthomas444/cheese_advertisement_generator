from dataset.dataset import CheeseDescriptionsDataset, CheeseDescriptionsTemplateDataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from tqdm import trange, tqdm
import torch
import random
import json
import argparse
from models.model_configs import *
from evaluation.all_metrics import run_evaluate
import json
import os


def driver(args, config, mode, from_template):

    print("Current device: ", torch.cuda.current_device())
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

    annot_file_test = config['data']['test']

    if not from_template:
        test_dataset = CheeseDescriptionsDataset(annotation_file=annot_file_test,
                                                loader_pipeline=pipeline)
    else:
        test_dataset = CheeseDescriptionsTemplateDataset(template_file=annot_file_test,
                                                         loader_pipeline=pipeline)

    bs = config['train']['batch_size']


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
        train=False
    )
    # model = torch.nn.DataParallel(model)
    model.to(device)

    model_forward_fnc = config_obj['model_forward']

    #### VALID #####
    test_bar = tqdm(
        test_loader,
        desc='Iter (loss=X.XXX)',
        disable=False
    )
    model.eval()
    with torch.no_grad():
        val_loss_list = []
        pred_list, inp_list, dec_list = [], [], []
        for test_step, test_batch in enumerate(test_bar):
            inp = [tens.to(device) for tens in test_batch]
            model_forward = model_forward_fnc(model=model,
                                              inp=inp)

            valid_loss = model_forward.loss.mean()

            val_loss_list.append(valid_loss)
            test_bar.set_description(
                'Iter (loss=%5.3f)' % valid_loss.item()
            )

            # prepare inputids and attention mask differently for "encoder-decoder" and "decoder-only" models
            if model.config.is_encoder_decoder:
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
                model.generate(
                    generate_input_ids,
                    attention_mask=attention_mask,
                    **config['generation_configs'][args.generation]), skip_special_tokens=True)
            # print('\n preds: ', preds)

            input = tokenizer.batch_decode(
                generate_input_ids, skip_special_tokens=True)

            dec_input = tokenizer.batch_decode(
                model_forward.labels, skip_special_tokens=True)

            pred_list.extend(preds)
            inp_list.extend(input)
            dec_list.extend(dec_input)

    # mode = "beam"
    # mode = "sampling_topk"
    # save the predictions in the directory where the checkpoint is saved
            
    if from_template:
        mode = mode+'_template'

    pred_save_dir = os.path.join(os.path.dirname(
        config['checkpoint']['config_path']), f"test_pred_{mode}.txt")
    gt_save_dir = os.path.join(os.path.dirname(
        config['checkpoint']['config_path']), f"test_gt_{mode}.txt")
    input_save_dir = os.path.join(os.path.dirname(
        config['checkpoint']['config_path']), f"input_gt_{mode}.txt")

    if from_template:
        # only save the pred file , no gt
        with open(input_save_dir, 'w', encoding='utf-8') as f2, open(pred_save_dir, 'w', encoding='utf-8') as f:
            for inp, pred in zip(inp_list, pred_list):
                f.write(pred+"\n###\n")
                f2.write(inp+"\n###\n")
    else:
        with open(pred_save_dir, 'w', encoding='utf-8') as f, open(gt_save_dir, 'w', encoding='utf-8') as f1, open(input_save_dir, 'w', encoding='utf-8') as f2:
            for inp, pred, gt in zip(inp_list, pred_list, dec_list):
                pred = pred.removeprefix(inp.strip())
                gt = gt.removeprefix(inp.strip())
                if not gt.strip():
                    continue
                f.write(pred+"\n###\n")
                f1.write(gt+"\n###\n")
                f2.write(inp+"\n###\n")

        # pipeline for evaluating

        results = run_evaluate(gt_file=gt_save_dir, pred_file=pred_save_dir)

        # results is array of [overall_json, rhet1_json, ..., rhetn_json]
        # save results
        for i, result_json in enumerate(results): 
            if i==0:
                result_save_file = os.path.join(os.path.dirname(
                    config['checkpoint']['config_path']), f"results_{mode}.json")
            else:
                result_save_file = os.path.join(os.path.dirname(
                    config['checkpoint']['config_path']), f"results_{mode}_{i}.json")
            with open(result_save_file, 'w') as f:
                json.dump(result_json, f)


def config_parser(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main():
    # argument parser stuff
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--from_template', action='store_true', help='To key values from template')
    parser.add_argument('--generation', type=str,
                        required=True,
                        help='Generate type (matched with config generate options eg: greedy)')

    args = parser.parse_args()
    config = config_parser(args.config)

    # invoke the generic training driver
    driver(args, config, mode=args.generation, from_template=args.from_template)


if __name__ == '__main__':
    main()
