import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Load datasets
rahul_df = pd.read_csv('rahul_reviews.csv')  # Adjust file path if necessary
modi_df = pd.read_csv('modi_reviews.csv')   # Adjust file path if necessary

# Add candidate labels
rahul_df['candidate'] = 'rahul'
modi_df['candidate'] = 'modi'

# Combine datasets
data = pd.concat([rahul_df, modi_df], ignore_index=True)

# Assuming datasets have columns 'Tweet' for comments and 'Sentiment' for positive/negative
def create_multiclass_label(row):
    if row['candidate'] == 'modi' and row['Sentiment'] == 'positive':
        return 0  # Modi Positive
    elif row['candidate'] == 'modi' and row['Sentiment'] == 'negative':
        return 1  # Modi Negative
    elif row['candidate'] == 'rahul' and row['Sentiment'] == 'positive':
        return 2  # Rahul Positive
    elif row['candidate'] == 'rahul' and row['Sentiment'] == 'negative':
        return 3  # Rahul Negative

data['multiclass_label'] = data.apply(create_multiclass_label, axis=1)

# Text preprocessing (optional)
def preprocess_text(text):
    # Add any text preprocessing steps like lowercasing, removing stopwords, etc.
    return text

data['Tweet'] = data['Tweet'].apply(preprocess_text)  # Use 'Tweet' column for preprocessing

# Split data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(data['Tweet'], data['multiclass_label'], test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Create a custom dataset class
class ElectionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create DataLoader objects
train_dataset = ElectionDataset(train_texts, train_labels)
test_dataset = ElectionDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define training parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optim.step()

# Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions, target_names=['modi_positive', 'modi_negative', 'rahul_positive', 'rahul_negative'])

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Predicting new data
def classify_comment(comment):
    comment = preprocess_text(comment)
    encoding = tokenizer(comment, return_tensors='pt', truncation=True, padding=True, max_length=512)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    labels = ['modi_positive', 'modi_negative', 'rahul_positive', 'rahul_negative']
    return labels[prediction]

# Example usage
new_comment = "Rahul's speech was very inspiring."
print(classify_comment(new_comment))
