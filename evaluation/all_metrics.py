import evaluate
import copy


def compute_metrics(results_dict, preds, gts):
    # compute rouge score
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=preds,
                                  references=gts)
    
    print("\n Rouge score: ", rouge_results)
    results_dict["rouge1"] = rouge_results['rouge1']
    results_dict["rouge2"] = rouge_results['rouge2']
    results_dict["rougeL"] = rouge_results['rougeL']

    # compute bleu score
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=preds,
                                references=gts)
    print("\n Bleu score: ", bleu_results)
    results_dict["bleu"] = bleu_results['bleu']

    # compute bert score
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=preds,
                                            references=gts, 
                                            lang="en")
    macro_f1 = sum(bertscore_results['f1'])/len(bertscore_results['f1'])
    macro_prec = sum(bertscore_results['precision'])/len(bertscore_results['precision'])
    macro_recall = sum(bertscore_results['recall'])/len(bertscore_results['recall'])

    print("\n Bert score (f1, p, r): ", macro_f1, macro_prec, macro_recall)
    results_dict["bert_score(f1)"] = macro_f1
    results_dict["bert_score(p)"] = macro_prec
    results_dict["bert_score(r)"] = macro_recall

    # compute meteor score
    meteor = evaluate.load('meteor')
    meteor_results = meteor.compute(predictions=preds,
                            references=gts)
    print('\n Meteor score: ', meteor_results)
    results_dict["meteor"] = meteor_results['meteor']
    return results_dict

def run_evaluate(gt_file, pred_file):
    results_dict = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "bleu": 0.0,
        "bert_score(f1)": 0.0,
        "bert_score(p)": 0.0,
        "bert_score(r)": 0.0,
        "meteor": 0.0
    }

    all_results = []
    with open(gt_file, 'r') as f, open(pred_file, 'r') as f1:
        gts, preds = f.read().split("\n###\n"), f1.read().split("\n###\n")

        gts = [x for x in gts if x]
        preds = [x for x in preds if x]
        print('\n Length of gts: ', len(gts))
        print('\n Length of preds: ', len(preds))


    gts_1 = [x for i,x in enumerate(gts) if i%6==0]
    gts_2 = [x for i,x in enumerate(gts) if i%6==1]
    gts_3 = [x for i,x in enumerate(gts) if i%6==2]
    gts_4 = [x for i,x in enumerate(gts) if i%6==3]
    gts_5 = [x for i,x in enumerate(gts) if i%6==4]
    gts_6 = [x for i,x in enumerate(gts) if i%6==5]

    preds_1 = [x for i,x in enumerate(preds) if i%6==0]
    preds_2 = [x for i,x in enumerate(preds) if i%6==1]
    preds_3 = [x for i,x in enumerate(preds) if i%6==2]
    preds_4 = [x for i,x in enumerate(preds) if i%6==3]
    preds_5 = [x for i,x in enumerate(preds) if i%6==4]
    preds_6 = [x for i,x in enumerate(preds) if i%6==5]

    print(len(gts_1), len(gts_2), len(gts_3), len(gts_4), len(gts_5), len(gts_6))

    for i in range(0,7):
        if i==0:
            rdict = compute_metrics(copy.deepcopy(results_dict), preds, gts)
        else:
            # for rhet units
            rdict = compute_metrics(copy.deepcopy(results_dict), eval(f"preds_{i}"), eval(f"gts_{i}"))
        all_results.append(rdict)
    
    print(all_results)
    return all_results


def main():
    # running test
    gt_file = 'checkpoints/flan_t5_base/test_gt_sampling_topk.txt'
    pred_file = 'checkpoints/flan_t5_base/test_pred_sampling_topk.txt'
    print(run_evaluate(gt_file, pred_file))

    
if __name__ == '__main__':
    main()
