from trainer import LambdaTrainer
from loader import PairDataset, Loader
import ir_datasets as irds
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from fire import Fire
import pandas as pd
import os
from os.path import join
import json
from trec23.pipelines.baselines import load_batchretrieve
from trec23 import CONFIG

def main(output_dir : str,
         batch_size : int,
         num_items : int,
         lr : float, 
         num_warmup_steps : int, 
         num_training_steps : int):
    
    os.makedirs(output_dir, exist_ok=True)
    
    query_lookup = pd.read_csv(join(CONFIG['MARCOv1_TEXT_PATH'], 'queries.tsv'), sep='\t', header=None, names=['query_id', 'query'], dtype={'query_id': str, 'query': str}).set_index('query_id')['query'].to_dict()
    doc_lookup = pd.read_csv(join(CONFIG['MARCOv1_TEXT_PATH'], 'docs.tsv'), sep='\t', header=None, names=['docno', 'text'], dtype={'docno': str, 'text': str}).set_index('docno')['text'].to_dict()

    train_pairs = pd.read_csv(join(CONFIG['MARCOv1_TEXT_PATH'], 'triples_small.tsv'), sep='\t', header=None, names=['query_id', 'doc_id_a', 'doc_id_b'], dtype={'query_id': str, 'doc_id_a': str, 'doc_id_b': str})
    train_pairs = train_pairs[['query_id', 'doc_id_a']].values.tolist()
    train_dataset = PairDataset(train_pairs, query_lookup, doc_lookup)
    train_loader = Loader(train_dataset, batch_size)
    
    model = ElectraForSequenceClassification.from_pretrained(CONFIG['ELECTRA_PATH'], num_labels=2)
    tokenizer = ElectraTokenizer.from_pretrained(CONFIG['ELECTRA_PATH'])

    retriever = load_batchretrieve(CONFIG['TERRIER_MARCOv1_PATH'], model="BM25")

    loss_kwargs = {
        'num_items': num_items,
        'batch_size': batch_size,
        'sigma': 1.,
        'ndcg_at': 50,
    }

    wrapper = LambdaTrainer(model, retriever, tokenizer, lr, batch_size, loss_kwargs=loss_kwargs)
    wrapper.train(train_loader, num_training_steps, num_warmup_steps)

    wrapper.model.save_pretrained(join(output_dir, 'model'))
    with open(join(output_dir, 'logs.json'), 'r') as f:
        json.dump(wrapper.logs, f)

if __name__ == '__main__':
    Fire(main)