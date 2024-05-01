from datasets import load_dataset
import re
import nltk
import random
from nltk import sent_tokenize, word_tokenize
import torch

###================================CONFIG
num_epochs = 5
num_easy_epochs = 2
warmup_steps = 0

scientific_raw = load_dataset('scientific_papers', 'arxiv', trust_remote_code=True,cache_dir='/home2/cgmj52/ResearchProject/Data/Datasets' )['train']
scientific_article = scientific_raw['article']


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

#have to use regex as NLTK sentence splitter doesn't work for some reason
# -*- coding: utf-8 -*-
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


scientific_preprocessed_2 = preprocess_data(scientific_article[:10])
scientific_article.clear()
from sklearn.model_selection import train_test_split
train_list, test_list = train_test_split(scientific_preprocessed_2, test_size=0.2, random_state=42)


from transformers import RobertaTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset

import itertools
import random
from torch.utils.data import Dataset, TensorDataset

class ClassifierDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=256, num_sentences=4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.easy_data = []


        for i in range(0, len(sentences), num_sentences):
            paragraph = sentences[i:i+num_sentences]
            shuffled_paragraph = random.sample(paragraph, len(paragraph))
            self.data.append(list(itertools.chain(*shuffled_paragraph)))
            self.labels.append(0)  
            for i in range(len(shuffled_paragraph)):
                if random.random() < 0.3:
                    words = word_tokenize(shuffled_paragraph[i])
                    shuffled_paragraph[i] = list(itertools.chain(*random.sample(words, len(words))))
            
            self.data.append(list(itertools.chain(*paragraph)))
            self.labels.append(1)  # Coherent paragraph
            self.easy_data.append(list(itertools.chain(*paragraph)))
            self.easy_data.append(list(itertools.chain(*shuffled_paragraph)))

            


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flattened_data = list(itertools.chain(*self.data[idx]))
        encoded_dict = self.tokenizer.encode_plus(
            flattened_data,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        flattened_easy_data = list(itertools.chain(*self.easy_data[idx]))
        easy_encoded_dict = self.tokenizer.encode_plus(
            flattened_easy_data,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_dict['input_ids'].squeeze(),
            'attention_mask': encoded_dict['attention_mask'].squeeze(),
            'easy_input_ids': easy_encoded_dict['input_ids'].squeeze(),
            'easy_attention_mask': easy_encoded_dict['attention_mask'].squeeze(),
            'labels': self.labels[idx]
        }


# Preprocess data
# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create dataset
train_dataset = ClassifierDataset(train_list, tokenizer)
test_dataset = ClassifierDataset(test_list, tokenizer)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)



import torch
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load pre-trained RoBERTa model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results/classifier',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=500,
    load_best_model_at_end=True,
    fp16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,

)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    precision = precision_score(labels, predictions.argmax(-1))
    recall = recall_score(labels, predictions.argmax(-1))
    f1 = f1_score(labels, predictions.argmax(-1))
    acc = accuracy_score(labels, predictions.argmax(-1))
    return {'precision': precision, 'recall': recall, 'f1': f1, 'acc':acc}

from sklearn.metrics import accuracy_score

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

from transformers import AdamW, get_linear_schedule_with_warmup

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Calculate the total number of training steps
total_steps = len(train_dataloader) * training_args.num_train_epochs

# Define the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
        input_ids, attention_mask, easy_input_ids, easy_attention_mask, label = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['easy_input_ids'].to(device),batch['attention_mask'].to(device), batch['labels'].to(device)
        if epoch < num_easy_epochs:
            outputs = model(easy_input_ids, attention_mask=easy_attention_mask, labels=label)
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)

        loss = outputs.loss
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
    
    # Evaluation on validation set
    model.eval()
    val_loss, val_accuracy = evaluate(model, test_dataloader, device)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
