from datasets import load_dataset
import torch
from utils import evaluate, preprocess_data
from data import ClassifierDataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, DataCollatorWithPadding
###================================CONFIG
num_epochs = 50
num_easy_epochs = num_epochs / 3
warmup_steps = 0

scientific_raw = load_dataset('scientific_papers', 'arxiv', trust_remote_code=True,cache_dir='/home2/cgmj52/ResearchProject/Data/Datasets' )['train']
scientific_article = scientific_raw['article']
scientific_preprocessed_2 = preprocess_data(scientific_article)
train_list, test_list = train_test_split(scientific_preprocessed_2, test_size=0.2, random_state=42)




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
best_accuracy = 0
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
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = model

best_model.save_pretrained(f'results/classifier/')
