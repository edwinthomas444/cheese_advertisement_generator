import torch
def bert_base_collator(batch):
    batch_t = [[] for _ in range(6)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        batch_t[0].append(torch.tensor(belem['input_ids'], dtype=torch.long))
        batch_t[1].append(torch.tensor(
            belem['attention_mask'], dtype=torch.long))
        batch_t[2].append(torch.tensor(
            belem['cross_attention_mask'], dtype=torch.long))
        batch_t[3].append(torch.tensor(
            belem['decoder_input_ids'], dtype=torch.long))
        batch_t[4].append(torch.tensor(
            belem['decoder_attention_mask'], dtype=torch.long))
        batch_t[5].append(torch.tensor(
            belem['labels'], dtype=torch.long))

    # stack tensors along batch dim
    for i in range(6):
        batch_t[i] = torch.stack(batch_t[i], dim=0)

    return batch_t

# gpt collator
def gpt_collator(batch):
    batch_t = [[] for _ in range(4)]
    # len batch is equal to batch size (based on Sampler used)
    for belem in batch:
        batch_t[0].append(torch.tensor(belem['input_ids'], dtype=torch.long))
        batch_t[1].append(torch.tensor(
            belem['attention_mask'], dtype=torch.long))
        batch_t[2].append(torch.tensor(
            belem['labels'], dtype=torch.long))
        batch_t[3].append(torch.tensor(
            belem['test_token_ids'], dtype=torch.long))

    # stack tensors along batch dim
    for i in range(3):
        # print("\n ",i,': ')
        # for x in batch_t[i]:
        #     print(x.shape)
        batch_t[i] = torch.stack(batch_t[i], dim=0)

    return batch_t