import evaluate

def main():
    gt_file = 'results/gt_gpt.txt'
    pred_file = 'results/preds_gpt.txt'

    with open(gt_file, 'r') as f, open(pred_file, 'r') as f1:
        gts, preds = f.read().split("###"), f1.read().split("###")

        gts = [x for x in gts if x]
        preds = [x for x in preds if x]


    # compute rouge score
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=preds,
                                  references=gts)
    print("\n Rouge score: ", rouge_results)
    
    # compute bleu score
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=preds,
                                references=gts)
    print("\n Bleu score: ", bleu_results)

    # compute bert score
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=preds,
                                            references=gts, 
                                            lang="en")
    macro_f1 = sum(bertscore_results['f1'])/len(bertscore_results['f1'])
    macro_prec = sum(bertscore_results['precision'])/len(bertscore_results['precision'])
    macro_recall = sum(bertscore_results['recall'])/len(bertscore_results['recall'])

    print("\n Bert score (f1, p, r): ", macro_f1, macro_prec, macro_recall)

    # compute meteor score
    meteor = evaluate.load('meteor')
    meteor_results = meteor.compute(predictions=preds,
                            references=gts)
    print('\n Meteor score: ', meteor_results)



if __name__ == '__main__':
    main()
