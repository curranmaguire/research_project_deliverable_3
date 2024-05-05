import re
import torch
from sklearn.metrics import accuracy_score

def clean_text(text):
    text = re.sub(r'\\[a-zA-Z]+(?![a-zA-Z])', '', text)
    
    # Remove specific LaTeX artifacts like @math1 and \cite{...}
    text = re.sub(r'@math\d+', '', text)  # Assuming @math1, @math2, etc.
    text = re.sub(r'@xmath\d+', '', text) 
    text = re.sub(r'\\cite\{[^}]*\}', '', text)  # Remove \cite{...}
    text = re.sub(r'&nbsp;','', text)
    text = re.sub(r'xcite','', text)
    
    # Normalize ellipses and multiple exclamation/question marks to a single instance
    text = re.sub(r'\.\.\.+', '…', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'([^\s\w,\.]|_)+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    return text


import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str):
    """
    https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences



def preprocess_data(dataset):
    
    
    # Tokenize into sentences
    
    # Create a list of split sentences
    out_df = []

    for paragraph in dataset:

        sentences = split_into_sentences(paragraph)

        [out_df.append(clean_text(sentence)) for sentence in sentences if clean_text(sentence) != None]

    return out_df


def generate_attention_score(tokens, model, tokenizer, max_length):
    # Tokenize the input tokens
    inputs = tokenizer.encode_plus(
        tokens,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move the inputs to the same device as the model
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Make predictions and get the attention weights from the encoder
    with torch.no_grad():
        outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        attentions = outputs.attentions

    # Extract the attention weights from the last layer
    last_layer_attentions = attentions[-1]

    # Average the attention weights across all heads
    averaged_attentions = last_layer_attentions.mean(dim=1).squeeze()

    # Normalize the attention scores
    normalized_attentions = averaged_attentions / averaged_attentions.sum(dim=-1, keepdim=True)

    # Get the attention scores for each token
    token_attention_scores = normalized_attentions[0, :len(tokens)]

    return token_attention_scores
# Generate attention scores for your dataset




def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, easy_input_ids, easy_attention_mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['easy_input_ids'].to(device),batch['attention_mask'].to(device), batch['labels'].to(device)
        

            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy
