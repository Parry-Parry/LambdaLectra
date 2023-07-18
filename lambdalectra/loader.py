import pandas as pd

class PairDataset:
    def __init__(self, pairs, corpus):
        docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()

        self.data = [{'query_id' : q, 'query' : queries[q], 'docno' : p, 'text' : docs[p]} for q, p in pairs]

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