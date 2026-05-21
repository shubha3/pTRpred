#generate tr data:


def extract_file(dataset_name, severity_level):
    files = f"{dataset_name}.xlsx_reform_severity_level{severity_level}.csv_rt_detect_result_window100.csv"
    p = DATA_DIR/files
    if p.exists():
        return p
    alt_name = dataset_name.replace('Soteria')

def normalize_dataset_name(name: str):
    if not name or not isinstance(name, str):
        return ""
    return re.sub(r"[\s\-]", "_", name.lower())

def find_rt_detect_file(dataset_name, severity_level):
    if not DATA_DIR.exists():
        return None
    base_norm = normalize_dataset(dataset_name)
    base_dash = re.sub(r"_+", "-", dataset_name.replace(' ', ""))
    list_files = DATA_DIR.glob("*rt_detect_result_window100.csv")
    for f in list_files:
        if f"severity_level{severity_level}" not in f.name:
            continue
        fnorm = normalize_dataset_name(f.stem.split('.xlsx')[0])
        if base_norm in fnorm or base_dash.lower() in f.name.lower():
            return f
    return None



#extract all of the temperature file series:
def load_temperature_series(path, max_length):
    if not DATA_DIR.exist():
        return None
    df = pd.read_csv(path)
    if 'temperature' not in df.columns:
        return np.full(max_length, np.nan)
    temperature = df['temperature'].astype(float).dropna().values
    if len(temperature) == 0:
        return np.full(max_length, np.nan)
    temperature = temperature[:max_length]
    if len(temperature) < max_length:
        temperature = np.pad(temperature, (0, max_length - len(temperature)),
            constant_values = temp[-1] if len(temp) > 0 else np.nan)
    return temp.astype(np.float32)


#Making the code more structured.
def load_labels_and_metadata(BATTERY_CLASS_PATH, label_col):
    df = pd.read_csv(BATTERY_CLASS_PATH)
    df = df[df[label_col].isin([0, 1])].copy()
    df.loc[i, 'severity'] = sevr
    return df


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)




def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds.append(logits.argmax(axis = 1).cpu().numpy())
            labels.append(y.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels






class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 512, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = 1.0/(10000 ** (torch.arange(0, d_model, 2).float()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))#the size here is (1, max_length, d_model) 
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CreateDataset(Dataset):
    def __init__(self, sequence, labels):
        self.X = sequences
        self.Y = labels
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, i):
        X = torch.from_numpy(self.X)
        Y = torch.tensor(self.y[i], dtype = torch.long)
        return X.unsqueeze(-1)#(using -1 is better.) -> the i~th position.







class CreateDataset(Dataset):
    def __init__(self, sequnences, labels):
        self.x = sequences
        self.y = labels
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i])
        y = torch.tensor(self.y[i], dtype = torch.long)


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]
    for i in range(X_train.shape[1]):
        col = 
        mn1 = np.nanmean(col)
        if np.isnan(mn1):
            mn1 = 15
        X_train[]

def run_kfold(X: np.ndarray, y: np.ndarray):
    device = torch.device('cuda' if torch.cuda.is_availabel() else 'cpu')
    skf = StratifiedKFold(n_splits = N_FOLDS, shuffle = True, random_state = RANDOM_STATE)
    fold_results = []
    all_preds, all_labels = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        for i in range(X_train.shape[1]):
            col = X_train[:, i]
            mn1 = np.nanmean(col)
            if np.isnan(mn1):
                mn1 = 15
            X_train[np.isnan(X_train[:, i]), i] = mn1
            X_val[np.isnan(X_val[:, i]), i] = mn1
        train_ds = CreateDataset(X_train, Y_train)
        val_ds = CreateDataset(X_val, Y_val)
        train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
        val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE)
        model = BatteryTransformer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0
        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, optimizer, criterion, device)
            p, l = evaluate(model, val_loader, device)
            acc = accuracy_score(l, p)
            if acc > best_acc:
                best_acc = acc
        p, l = evaluate(model, val_loader, device)
        fold_results.append({
        	'fold': fold,
        	'accuracy': accuracy_score(l, p),
        	'precision': precision_recall_fscore_support(l, p, average = 'binary')[0],
        	'recall': precision_recall_fscore_support(l, p, average = 'binary')[1],
        	'f1': precision_recall_fscore_support(l, p, average = 'binary')[2]
        	})
        all_preds.append(p)
        all_labels.append(l)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
    'fold_results': fold_resuts,
    'overall_accuracy': accuracy_score(all_labels, all_preds),
    'confusion_matrix': confusion_matrix(all_labels, all_preds),
    'precision_recall_f1': precision_recall_fscore_support(all_labels, all_preds,
        average = 'binary')
    }


#run for each of the dataset:

#Writing the args here:

N_EPOCH = 30
BATCH_SIZE = 32
MAX_LENGTH = 128
N_FOLDS = 5



class EncoderClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(Fin, args.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model = d_model,
            nhead = max(1, min(args.n_head, args.d_model//2)),
            dim_feedforward = 4 * args.d_model,
            dropout = args.dropout,
            batch_first = True,
            activation = 'gelu',
            norm_first = True
        )#B, n_token, d_model
        self.enc = nn.TransformerEncoder(enc_layers, num_layers = args.nlayers)
        self.cls = nn.Linear(args.d_model, 2)
    def forward(self, x):
        z = self.proj(x)
        h = self.enc(z)
        p = torch.mean(h, dim = 1)
        return self.cls(p)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncoderClassifier().to(device)
opt = torch.optim.AdamW(model.parameters(), lr = args.lr)


#Self-Pipeline:
def extract_embedding(encoder, sequences):
    device = 'cuda' if cuda.is_available() else 'cpu'
    encoder.val()
    with torch.no_grad():
        X = torch.tensor(sequences, dtype = torch.float32, device = device).unsqueeze(-1)
        out = encoder(X)#(n_seq, input_window_size, d_model) -> (, d_model)
        pooled = out.mean(dim = (0,1))
    return pooled.cpu().numpy()



#arg
#Extract all of the data here:
import os
import numpy as np
import pandas as np
import warnings
import sys
import argparse
from pathlib import Path
from TR_binary_transformer import *

os.chdir("/Users/heqiaoruan/Documents/Github/pTRpred")
PATH = Path("datasets/NREL/Battery_Cells/processed_data")
file_list = PATH.glob("*.csv")
#Reading all of the dataset:
file_battery = [file for file in file_list if "severity_level" in str(file)]
battery_class = pd.read_csv("datasets/NREL/Battery_cells/processed_data/battery_class_results.csv")
MAX_EXTENT = 1500

#Identify the time-point that is starting to become significant:


for file in file_battery:
    battery_data = pd.read_csv(file)
    battery_name = str(file).split('/')[-1].split('.')[0]
    label = battery_class[battery_class['dataset_name'] == battery_name]['Label'].item()
    X = battery_data[['temperature', 'load', 
        'voltage', 'force']].iloc[:min(MAX_EXTENT, battery_data.shape[0]), :].to_numpy()
    y = 
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    #Run Kfold for the DataLoader()

    
    
    


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import Pipeline

dfX = np.random.random((10000, 100))
Y = np.random.random((10000, ))
X_train, Y_train, X_test, Y_test = train_test_split(dfX, Y, test_size = 0.25)

pipeline = Pipeline(steps = [
    ('scaler', StandardScaler()),
    ('RF', RandomForestRegressor(random_state = 42))
])

param_grid = {
    'RF__n_estimators': [100, 150, 200],
    'RF__max_depth': [2, 4, 6, 8]
}
grid_search = GridSearchCV(pipeline, param_grid, cv = 10)
grid_search.fit(dfX, Y)
























