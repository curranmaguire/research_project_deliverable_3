import torch
from transformers import BartForConditionalGeneration, RobertaForSequenceClassification, RobertaTokenizer, BartTokenizer
from utils import generate_attention_score, split_into_sentences

num_itterations = 3


def inference(input_sentences, coherence_classifier, bart_model, classifier_tokenizer, bart_tokenizer, lambda_threshold=0.9, window_size=2, max_length=128):
    input_sentences = split_into_sentences(input_sentences)
    num_sentences = len(input_sentences)
    coherent_sentences = []
    incoherent_span = []
    after_incoherence = []
    i=0
    span_created=False
    for i in range(num_sentences - window_size):
        
        inputs = classifier_tokenizer.encode_plus(
        ' '.join(input_sentences[i:i+window_size]),
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(coherence_classifier.device)
        attention_mask = inputs['attention_mask'].to(coherence_classifier.device)
        outputs = coherence_classifier(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        print(f'prediction from classifier is {int(preds[0])} on span {" ".join(input_sentences[i:i+window_size])}')


        if int(preds[0]) == 1:
            #coherent so need to check if previous incoherence
            if not span_created:
                i += 1
                coherent_sentences.append(input_sentences[i])
                continue
            else:
                #incoherent span has ended and assign the rest of the text to the post span list
                

                after_incoherence = input_sentences[i:]
                break  
        else:
            #need to add the span to the incoherent list
            if i + window_size == num_sentences:
                incoherent_span.append(input_sentences[i:])
            incoherent_span.append(input_sentences[i])
            i+=1
            #incoherent span so we need to switch the incoherence marker on and continue
    
    
    #after this loop we have a span of incoherent sentences
    #we now pass this span through the classifier and then pass the attention to Bart to be masked and inferred
    attention_scores = generate_attention_score(' '.join(incoherent_span), coherence_classifier, classifier_tokenizer, max_length)
    masked_tokens = []
    for token_id, score in zip(classifier_tokenizer.encode(' '.join(incoherent_span)), attention_scores):
        if score > lambda_threshold:
            masked_tokens.append(classifier_tokenizer.mask_token)
        else:
            masked_tokens.append(classifier_tokenizer.convert_ids_to_tokens([token_id])[0])

    if len(incoherent_span) > 0:
        masked_paragraph = classifier_tokenizer.convert_tokens_to_string(masked_tokens)
        encoded_bart = bart_tokenizer.encode_plus(
            masked_paragraph,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        output = bart_model.generate(encoded_bart['input_ids'], max_length=bart_model.config.max_length, num_beams=4, early_stopping=True)
        coherent_output = bart_tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        print('No incoherence!')
        return ' '.join(input_sentences)
    return ' '.join(coherent_sentences) + coherent_output + ' '.join(after_incoherence)


coherence_classifier =  RobertaForSequenceClassification.from_pretrained('results/classifier/') 
bart_model = BartForConditionalGeneration.from_pretrained('results/bart/')
classifier_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

output = []
with open('input.txt', 'r') as f:
    for i, line in enumerate(f):
        for i in range(num_itterations):
            new_example = inference(line, coherence_classifier, bart_model, classifier_tokenizer, bart_tokenizer)
            if new_example == example:
                break
            else:
                example = new_example
        output.append(example)

with open('inference_output.txt', 'w') as f:
    for i in output:
        f.write(i + '\n')
