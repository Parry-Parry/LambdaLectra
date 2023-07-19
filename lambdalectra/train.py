from .trainer import LambdaTrainer
from .loader import PairDataset, Loader
import ir_datasets as irds
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from fire import Fire
import pandas as pd
from os.path import join
import json
from trec23.pipelines.baselines import load_splade, load_pisa
from trec23 import CONFIG

def main(model_name_or_path, 
         dataset, 
         dataset_dir,
         output_dir,
         batch_size,
         lr, 
         num_warmup_steps, 
         num_training_steps,
         mode = 'splade',
         splade_path = None,
         pisa_path = None):
    
    ds = irds.load(dataset)

    train_pairs = pd.read_csv(dataset_dir, header=None, names=['query_id', 'docno'])
    train_dataset = PairDataset(train_pairs, ds)
    train_loader = Loader(train_dataset, 1)
    
    model = ElectraForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    tokenizer = ElectraTokenizer.from_pretrained(model_name_or_path)

    if mode == 'splade':
        retriever = load_splade(splade_path if splade_path else CONFIG['SPLADE_MARCOv2_PATH'])
    else:
        retriever = load_pisa(pisa_path if pisa_path else CONFIG['PISA_MARCOv2_PATH'])

    loss_kwargs = {
        'num_items': batch_size-1,
        'batch_size': batch_size,
        'sigma': 1.,
        'ndcg_at': 10,
    }

    wrapper = LambdaTrainer(model, retriever, tokenizer, lr, batch_size, loss_kwargs=loss_kwargs)
    wrapper.train(train_loader, num_training_steps, num_warmup_steps)

    wrapper.model.save_pretrained(join(output_dir, 'model'))
    with open(join(output_dir, 'logs.json'), 'r') as f:
        json.dump(wrapper.logs, f)

if __name__ == '__main__':
    Fire(main)