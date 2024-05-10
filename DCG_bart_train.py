from datasets import load_dataset
import torch
from utils import preprocess_data
from data import BartDataset
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BartTokenizer, DataCollatorWithPadding, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
###================================CONFIG
num_epochs = 50
max_length = 514 #this is because the GPUs I have access to would OOM if I assigned greater. It is recommended to make this the maximum of 514
warmup_steps = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



scientific_raw = load_dataset('scientific_papers', 'arxiv', trust_remote_code=True,cache_dir='/home2/cgmj52/ResearchProject/Data/Datasets' )['train']
scientific_article = scientific_raw['article']

scientific_preprocessed_2 = preprocess_data(scientific_article)

train_list, test_list = train_test_split(scientific_preprocessed_2, test_size=0.2, random_state=42)

# Load the trained classifier model
classifier_model = RobertaForSequenceClassification.from_pretrained('results/classifier').to(device)
classifier_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

classifier_model.roberta.embeddings.position_embeddings.weight.requires_grad = False
classifier_model.roberta.embeddings.position_embeddings.eval()
# Set the model to evaluation mode
classifier_model.eval()




print(f'starting dataset creation...')
# Preprocess data
# Load RoBERTa tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
print('completed dataset creation now training.')
# Create dataset
train_dataset = BartDataset(train_list, classifier_tokenizer, classifier_model, tokenizer)
test_dataset = BartDataset(test_list, classifier_tokenizer, classifier_model,  tokenizer)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=data_collator)





# Load pre-trained BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')




# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Calculate the total number of training steps
total_steps = len(train_dataloader) * num_epochs

# Define the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

model.to(device)
best_loss = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    print(f'starting epoch {epoch}...')
    
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
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
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target)
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(test_dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        best_model = model

best_model.save_pretrained(f'results/bart/')
test_dataloader.save_input('inputs.txt')
