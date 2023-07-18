from .trainer import LambdaWrapper
from .loader import PairDataset, Loader
import ir_datasets as irds
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from fire import Fire
import pandas as pd
from os.path import join
import json

def main(model_name_or_path, 
         dataset, 
         dataset_dir,
         output_dir,
         batch_size,
         lr, 
         num_warmup_steps, 
         num_training_steps):
    
    ds = irds.load(dataset)

    train_pairs = pd.read_csv(dataset_dir, header=None, names=['query_id', 'docno'])
    train_dataset = PairDataset(train_pairs, ds)
    train_loader = Loader(train_dataset, 1)
    
    model = ElectraForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

    retriever = None 

    loss_kwargs = {
    }

    wrapper = LambdaWrapper(model, retriever, tokenizer, lr, batch_size, num_neg=batch_size-1, loss_kwargs=loss_kwargs)
    wrapper.train(train_loader, num_training_steps, num_warmup_steps)

    wrapper.model.save_pretrained(join(output_dir, 'model'))
    with open(join(output_dir, 'logs.json'), 'r') as f:
        json.dump(wrapper.logs, f)

if __name__ == '__main__':
    Fire(main)