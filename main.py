import pandas as pd
import numpy as np
import re

import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import joblib


# Загрузка датасета

df_main = pd.read_csv("input_dataframe/payments_main.tsv", sep='\t', names=["id", "date", "amount", "text"])
df_main = df_main.drop(columns=["date"])

# Предобработка данных


def parse_amount(amount):
    if not isinstance(amount, str):
        return None

    amount = amount.replace(" ", "")
    amount = amount.replace("-", ".")

    if ',' in amount and '.' in amount:
        # Mixed format (e.g., 14.000,00 or 14,000.00)
        if amount.index(',') > amount.index('.'):
            # Format like 14.000,00 (dot as thousand separator, comma as decimal separator)
            amount = amount.replace('.', '').replace(',', '.')
        else:
            # Format like 14,000.00 (comma as thousand separator, dot as decimal separator)
            amount = amount.replace(',', '')
    elif ',' in amount:
        if amount.index(',') < len(amount)-3:
            amount = amount.replace(',', '')
        else:
            amount = amount.replace(',', '.')
    elif '.' in amount:
        # Format like 14.000 or 14000.00 (dot as thousand separator or decimal point)
        amount = re.sub(r'\.(?=\d{3}(?!\d))', '', amount)  # Remove dots used as thousand separator

    # Convert to float
    try:
        return float(amount)
    except ValueError:
        return None


def clean_meta_info(text):
    # Преобразовать в нижний регистр
    text = text.lower()

    # Удалить номера договоров: "№452", "№E01368" и подобное
    text = re.sub(r"№\S+", " ", text)
    # Удалить даты в форматах: "17.03.2024", "17-03-24", "17/03/2024", "17.03.2024г", "01/01/2024г"
    text = re.sub(r"\d{1,2}[-./]\d{1,2}[-./]\d{2,4}\s?г?\.?", " ", text)
    # Удалить год в формате: "2024г"
    text = re.sub(r"\b\d{4}\s?г\.?\b", " ", text)
    # Удалить названия месяцев
    text = re.sub(r'\b(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b', ' ', text)
    # Удалить суммы: "100 000.50", "2400000,00", "2400000.00", "100000-50"
    text = re.sub(r"\b\d{1,3}([ .,-]?\d{3})*(\.\d+|,\d+|-?\d+)?\b", " ", text)
    # Удалить символы валют
    text = re.sub(r'\b(?:₽|доллар(?:ов|а)?|USD|RUB|руб(?:.|ль|ля|лей)?)\b', ' ', text, flags=re.IGNORECASE)

    # Удалить служебные слова без информации: "на сумму", "от", "в т.ч."
    text = re.sub(r"\b(?:сумма|на сумму|от|и т\.д\.|в т\.ч\.?|в том числе|г\.)\b", " ", text, flags=re.IGNORECASE)


    # Удалить отдельно стоящие дефисы
    text = re.sub(r'-+', '-', text)
    # Удалить процентные знаки
    text = re.sub(r'%', ' ', text)
    # Удалить скобки
    text = re.sub(r'[()]', ' ', text)
    # Удалить лишние запятые
    text = re.sub(r"\s{2,}", " ", text).strip()

    text = text.replace(".", " ")
    text = text.replace("/", " ")

    # Удалить отдельно стоящие дефисы
    text = re.sub(r'(?<!\w)-(?!\w)|-+(?=\s|$)', '', text)
    # Удалить слова короче 3 символов
    text = re.sub(r'\s+\w{1,2}\s+', ' ', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text)

    return text

try:
    df_main['amount'] = df_main['amount'].apply(parse_amount)
    df_main['cleaned_text'] = df_main['text'].apply(clean_meta_info)
except:
    pass


def preprocess(text, stop_words, punctuation_marks, morph):
    tokens = word_tokenize(text.lower())
    preprocessed_text = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token[0] == '-' or token[-1] == '-':
            continue
        if token not in punctuation_marks:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in stop_words:
                preprocessed_text.append(lemma)
    return preprocessed_text

punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...']
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()

df_main['preprocessed_text'] = df_main.apply(
    lambda row: preprocess(row['cleaned_text'], punctuation_marks, stop_words, morph), axis=1
)

preprocessed_df_main = df_main[['id', 'amount', 'preprocessed_text']]

texts = [" ".join(text) for text in preprocessed_df_main['preprocessed_text']]
preprocessed_df_main['preprocessed_text'] = texts

# Загрузка модели и прогон данных

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Определяем Dataset

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# 2. Инициализация токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
model_rubert = AutoModel.from_pretrained('cointegrated/rubert-tiny').to(device)

texts = preprocessed_df_main['preprocessed_text']

# 3. Создаём Dataset и DataLoader
text_dataset = TextDataset(texts)
batch_size = 512
data_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False)

# 4. Обработка данных батчами
embeddings = []

model_rubert.eval()
with torch.no_grad():
    for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader), ncols=100):
        encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        input_ids = encoded_inputs['input_ids'].to(device)
        attention_mask = encoded_inputs['attention_mask'].to(device)

        outputs = model_rubert(input_ids=input_ids, attention_mask=attention_mask)

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeddings)
final_embeddings = torch.cat(embeddings, dim=0)
final_embeddings_list = [emb.cpu().numpy() for emb in final_embeddings]
preprocessed_df_main['text_embed'] = final_embeddings_list

scaler = MinMaxScaler()
preprocessed_df_main['amount_normalized'] = scaler.fit_transform(preprocessed_df_main[['amount']])

hidden_size = 768


class AmountProcessor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AmountProcessor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

amount_processor = AmountProcessor(input_dim=1, output_dim=hidden_size)
text_embeddings_array = np.stack(preprocessed_df_main['text_embed'].values)  # [num_samples, hidden_size]
amount_normalized_array = preprocessed_df_main['amount_normalized'].values.reshape(-1, 1)  # [num_samples, 1]
amount_processor = amount_processor.to(device)
text_embeddings_tensor = torch.tensor(text_embeddings_array, dtype=torch.float32).to(device)
amount_normalized_tensor = torch.tensor(amount_normalized_array, dtype=torch.float32).to(device)
processed_amount = amount_processor(amount_normalized_tensor)  # [num_samples, hidden_size]
combined_features = torch.cat((text_embeddings_tensor, processed_amount), dim=1)  # [num_samples, 2 * hidden_size]

combined_features_cpu = combined_features.cpu().detach().numpy()
combined_features_list = [feature.tolist() for feature in combined_features_cpu]
preprocessed_df_main['combined_features'] = combined_features_list

preprocessed_df_main = preprocessed_df_main[['id', 'amount', 'preprocessed_text', 'combined_features']]

# Загрузка обученной модели

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_prob=0.3):
        super(ClassificationModel, self).__init__()
        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

categories = ['BANK_SERVICE', 'FOOD_GOODS', 'LEASING', 'LOAN', 'NON_FOOD_GOODS', 'NOT_CLASSIFIED', 'REALE_STATE', 'SERVICE', 'TAX']
num_classes = len(categories)

X_tensor = torch.tensor(preprocessed_df_main['combined_features'], dtype=torch.float32).to(device)
input_dim = X_tensor.shape[1]
hidden_dims = [1024, 512, 256]
model = ClassificationModel(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model_weights/model.pth', map_location=torch.device('cpu')))
model.to(device)

# Предсказание

model.eval()
with torch.no_grad():
    outputs = model(X_tensor)
_, predictions = outputs.max(1)
label_encoder = joblib.load('model_weights/label_encoder.pkl')

predicted_labels = label_encoder.inverse_transform(predictions.cpu().numpy())
preprocessed_df_main['predicted_label'] = predicted_labels
output_df = preprocessed_df_main[['id', 'preprocessed_text', 'predicted_label']]

output_df = output_df[['id', 'predicted_label']]

output_df.to_csv('output_dataframe/output.tsv', sep='\t', index=False, header=False)