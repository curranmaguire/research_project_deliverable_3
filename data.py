
import itertools
import random
from nltk import word_tokenize
from utils import generate_attention_score
from torch.utils.data import Dataset

class BartDataset(Dataset):
    def __init__(self, sentences,classifier_tokenizer, classifier, tokenizer, max_length=514, num_sentences=4, lambda_threshold = 0.5):
        self.tokenizer = tokenizer
        self.classifier_tokenizer = classifier_tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.easy_data = []


        for i in range(0, len(sentences), num_sentences):
            paragraph = sentences[i:i+num_sentences]
            shuffled_paragraph = random.sample(paragraph, len(paragraph))
            for i in range(len(shuffled_paragraph)):
                if random.random() < 0.3:
                    words = word_tokenize(shuffled_paragraph[i])
                    shuffled_paragraph[i] = ' '.join(random.sample(words, len(words)))

            # Tokenize the original and shuffled paragraphs
            original_tokens = self.classifier_tokenizer.tokenize(' '.join(paragraph))
            shuffled_tokens = self.classifier_tokenizer.tokenize(' '.join(shuffled_paragraph))
            masked_tokens = []
            shuffled_masked_tokens = []
            shuffled_attention_scores = generate_attention_score(' '.join(shuffled_paragraph), classifier, classifier_tokenizer, max_length)
            attention_scores = generate_attention_score(' '.join(paragraph), classifier, classifier_tokenizer, max_length)
            for token, score in zip(shuffled_tokens, shuffled_attention_scores[i:i+num_sentences]):
                if score > lambda_threshold:
                    shuffled_masked_tokens.append(self.classifier_tokenizer.mask_token)
                else:
                    shuffled_masked_tokens.append(token)
            for token, score in zip(original_tokens, attention_scores[i:i+num_sentences]):
                if score > lambda_threshold:
                    masked_tokens.append(self.classifier_tokenizer.mask_token)
                else:
                    masked_tokens.append(token)
            # Convert the masked tokens back to text
            masked_paragraph = self.classifier_tokenizer.convert_tokens_to_string(masked_tokens)
            shuffled_masked_paragraph = self.classifier_tokenizer.convert_tokens_to_string(shuffled_masked_tokens)
            # Tokenize and truncate the masked paragraph
            masked_input = self.tokenizer.encode_plus(
                masked_paragraph,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            shuffled_masked_input = self.tokenizer.encode_plus(
                shuffled_masked_paragraph,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Tokenize and truncate the original paragraph as the target
            target = self.tokenizer.encode_plus(
                paragraph,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.data.append({
            'input_ids': masked_input['input_ids'].squeeze(),
            'attention_mask': masked_input['attention_mask'].squeeze(),
            'target': target['input_ids'].squeeze()
            })
            self.data.append({
            'input_ids': shuffled_masked_input['input_ids'].squeeze(),
            'attention_mask': shuffled_masked_input['attention_mask'].squeeze(),
            'target': target['input_ids'].squeeze()
            })
            


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return {
            'input_ids': self.data[idx]['input_ids'].squeeze(),
            'attention_mask': self.data[idx]['attention_mask'].squeeze(),
            'target': self.data[idx]['target'].squeeze()
        }
    
    
    def save_input(self, file_path):
        with open(file_path, 'w') as file:
            for item in self.data:
                input_text = self.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
                file.write(f"{input_text}\n")
        with open('targets.txt', 'w') as file:
            for item in self.data:
                target_text = self.tokenizer.decode(item['target'], skip_special_tokens=True)
                file.write(f"{target_text}\n")

class ClassifierDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=514, num_sentences=4):
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

