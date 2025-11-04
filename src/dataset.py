import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_or_build_tokenizer(config, data, lang):
    tokenizer_path = os.path.join(config['tokenizer_path'], f"tokenizer_{lang}.json")
    if not os.path.exists(tokenizer_path):
        print(f"Building tokenizer for {lang}...")
        os.makedirs(config['tokenizer_path'], exist_ok=True)
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], vocab_size=config['vocab_size'])
        
        def batch_iterator(batch_size=1000):
            for i in range(0, len(data), batch_size):
                yield [item[lang] for item in data[i : i + batch_size]['translation']]

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        print(f"Loading tokenizer for {lang}...")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, hf_dataset, src_tokenizer, tgt_tokenizer, config):
        self.hf_dataset = hf_dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = config['src_lang']
        self.tgt_lang = config['tgt_lang']
        self.max_seq_len = config['max_seq_len']
        
        self.sos_token_id = self.tgt_tokenizer.token_to_id("[SOS]")
        self.eos_token_id = self.tgt_tokenizer.token_to_id("[EOS]")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        pair = self.hf_dataset[idx]['translation']
        src_text = pair[self.src_lang]
        tgt_text = pair[self.tgt_lang]
        
        src_ids = self.src_tokenizer.encode(src_text).ids
        tgt_ids = self.tgt_tokenizer.encode(tgt_text).ids

        src_ids = src_ids[:self.max_seq_len-2] # Truncate if necessary
        tgt_ids = tgt_ids[:self.max_seq_len-1]

        enc_input = torch.tensor(src_ids)
        dec_input = torch.tensor([self.sos_token_id] + tgt_ids)
        label = torch.tensor(tgt_ids + [self.eos_token_id])

        return enc_input, dec_input, label

def create_dataloaders(config):
    # 1. Load dataset from Hugging Face
    raw_datasets = load_dataset(config['dataset_name'], config['dataset_config'])
    train_data = raw_datasets['train']
    val_data = raw_datasets['validation']

    # 2. Build or load tokenizers
    src_tokenizer = get_or_build_tokenizer(config, train_data, config['src_lang'])
    tgt_tokenizer = get_or_build_tokenizer(config, train_data, config['tgt_lang'])
    
    # 3. Create Dataset objects
    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, config)
    val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, config)
    
    pad_idx = src_tokenizer.token_to_id("[PAD]")

    def collate_fn(batch):
        src_batch, dec_in_batch, label_batch = [], [], []
        for src_sample, dec_in_sample, label_sample in batch:
            src_batch.append(src_sample)
            dec_in_batch.append(dec_in_sample)
            label_batch.append(label_sample)
            
        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
        dec_in_batch = pad_sequence(dec_in_batch, batch_first=True, padding_value=pad_idx)
        label_batch = pad_sequence(label_batch, batch_first=True, padding_value=pad_idx)
        
        return src_batch, dec_in_batch, label_batch

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, src_tokenizer, tgt_tokenizer