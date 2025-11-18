from torch.utils.data import Dataset
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(args, split):
    df = pd.read_csv(f"{args.data_root}/{split}.csv")
    texts = df['text'].astype(str).values.tolist()
    # 测试集test.csv没有target列
    if 'target' in df.columns:
        labels = df['target'].values.tolist()
    else:
        labels = None
    return texts, labels

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = data[0]
        self.labels = data[1] if len(data) > 1 else None
        self.is_test = is_test
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        text = str(self.texts[index])
        source = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.max_length,
            # pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        data_sample = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
        }
        # 仅当不是测试集且 labels 可用时附上 labels
        if (not self.is_test) and (self.labels is not None):
            label = self.labels[index]
            data_sample["labels"] = torch.tensor(label).squeeze()
        return data_sample
    
if __name__ == "__main__":
    df = pd.read_csv('data/train_src.csv')
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['target'], random_state=42)
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
