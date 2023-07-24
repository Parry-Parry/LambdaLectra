import pandas as pd

class PairDataset:
    def __init__(self, pairs, query_lookup, doc_lookup) -> None:
        self.data = [{'qid' : q, 'query' : query_lookup[q], 'docno' : p, 'text' : doc_lookup[p]} for q, p in pairs]

    def __len__(self):
        return len(self.data)
    
    def get_items(self, idx):
        return self.data[idx]

class Loader:
    def __init__(self, dataset, batch_size) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)

    def get_batch(self, idx):
        return pd.DataFrame.from_records([self.dataset.get_items(j) for j in range(idx * self.batch_size, (idx + 1) * self.batch_size)])