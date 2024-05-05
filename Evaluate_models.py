import lmppl
from evaluate import load
from nltk.translate.bleu_score import corpus_bleu
bertscore = load("bertscore")

scorer = lmppl.LM('gpt2')




with open('DCG_input.txt') as f:
    source = f.readlines()
with open('DCG_output.txt') as f:
    output = f.readlines()
with open('DCG_original.txt') as f:
    original = f.readlines()
ppl = scorer.get_perplexity(output)
print('ppl score of the test data is:\n')
print(list(zip(output, ppl)))
average_ppl = sum(ppl) / len(ppl)
print(f'average PPL score is {average_ppl}')
bert_metric = bertscore.compute(predictions=output, references=original, lang="en")

print(f'bert score for DCG is {bert_metric}')

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

model_path = "/home2/cgmj52/ResearchProject/Classifier/Bert_classifier/checkpoint-2500"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
def evaluate_style_accuracy(texts, model, tokenizer):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_samples = len(texts)
    correct_predictions = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        if predicted_label == 1:  # Assuming label 1 represents the desired style
            correct_predictions += 1

    style_accuracy = correct_predictions / total_samples
    return style_accuracy


style_accuracy = evaluate_style_accuracy(output, model, tokenizer)
print(f"Style accuracy for DCG output: {style_accuracy:.4f}")

original_style_accuracy = evaluate_style_accuracy(source,model, tokenizer)
print(f'Style accuracy for DCG original text: {original_style_accuracy:.4f}')

source_bleu = corpus_bleu(original, output)
