import evaluate

def run_evaluate(gt_file, pred_file):
    with open(gt_file, 'r') as f, open(pred_file, 'r') as f1:
        gts, preds = f.read().split("\n###\n"), f1.read().split("\n###\n")

        gts = [x for x in gts if x]
        preds = [x for x in preds if x]


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


def main():
    # running test
    gt_file = 'checkpoints/transformers_bert_base/test_gt.txt'
    pred_file = 'checkpoints/transformers_bert_base/test_pred.txt'
    print(run_evaluate(gt_file, pred_file))

    
if __name__ == '__main__':
    main()
