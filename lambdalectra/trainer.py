import pyterrier as pt
if not pt.started():
    pt.init()
from collections import defaultdict
import time
import torch 
import numpy as np
import ir_datasets

from transformers import AdamW, get_linear_schedule_with_warmup

from loss import LambdaRankLoss, LambdaRankLossFn

RND = 42

class LambdaTrainer:
    def __init__(self,
                 model,
                 retriever,
                 tokenizer,
                 lr,
                 batch_size = 16,
                 cutoff = 1000,
                 loss_kwargs = {},) -> None:

        self.logs = {
            'batch_size': batch_size,
            'lr': lr,
            'loss' : defaultdict(list),
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_component = LambdaRankLoss(**loss_kwargs)
        self.reshape = lambda x : x.view(self.loss_component.batch_size, self.loss_component.num_items)

        index = pt.get_dataset('irds:msmarco-passage/train/triples-small')
        self.retrieve = retriever % cutoff >> pt.text.get_text(index, 'text')
        self.cutoff = cutoff
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        self.batch_size = batch_size

    def process_logits(self, logits):
        probs = logits.float().softmax(dim=-1)[:, 0]
        return self.reshape(probs)

    def ranking_to_batch(self, ranking):
        queries = []
        docs = []
        labels = []

        for row in ranking.itertuples():
            queries.append(row.query)
            docs.append(row.text)
            labels.append(row.label)
        
        return (queries, docs), torch.tensor(np.array(labels))
    
    def first_pass(self, batch):
        results = self.retrieve.transform(batch[['qid', 'query']]) 
        results.drop(['score', 'rank'], axis=1, inplace=True)

        results = results.groupby('qid').apply(lambda x : x.iloc[np.linspace(0, len(x), self.loss_component.num_items - 1, endpoint=False, dtype=int)]).reset_index(drop=True)

        results['label'] = [np.array([0, 1])] * len(results)
        lookup = batch.set_index('qid')[['docno', 'text']].to_dict('index')
        # collapse results to a qid, query and a list of tuples of docno, text and label
        results = results.groupby(['qid', 'query'])[['docno', 'text', 'label']].apply(lambda x: list(x.itertuples(index=False, name=None))).reset_index()
        # rename column 0 to documents
        results.rename(columns={0: 'documents'}, inplace=True)
        for i, row in enumerate(results.itertuples()):
            vals = lookup[row.qid]
            tmp = getattr(row, 'documents')
            tmp.insert(0, (vals['docno'], vals['text'], np.array([1, 0])))
            results.at[i, 'documents'] = tmp

        # unfold the list of tuples into a dataframe with new columns docno, text and label
        results = results.explode('documents')
        results['docno'] = results['documents'].apply(lambda x : x[0])
        results['text'] = results['documents'].apply(lambda x : x[1])
        results['label'] = results['documents'].apply(lambda x : x[2])
        results.drop('documents', axis=1, inplace=True)
        results = results.reset_index(drop=True)

        return self.ranking_to_batch(results)

    def main_loop(self, batch):
        batch, labels = self.first_pass(batch)
        inputs = self.tokenizer(batch[0], batch[1], padding=True, truncation=True, return_tensors='pt')
        ids = inputs['input_ids'].to(self.device)
        mask = inputs['attention_mask'].to(self.device)
        labels = labels.to(self.device)
        y_pred = self.process_logits(self.model(ids, attention_mask=mask).logits)
        y_true = self.process_logits(labels)
        loss = LambdaRankLossFn.forward(, , self.loss_component)  
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def train(self, train_loader, num_training_steps, warmup_steps=0):
        torch.manual_seed(RND)
        _logger = ir_datasets.log.easy()
        self.train_loader = train_loader   
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=warmup_steps, 
                                                         num_training_steps=num_training_steps)
        
        start = time.time()
        with _logger.pbar_raw(desc=f'train', total=num_training_steps // self.batch_size) as pbar:
            for i in range(num_training_steps // self.batch_size):
                batch = train_loader.get_batch(i)
                loss = self.main_loop(batch)

                self.logs['loss']['train'].append(loss.item())
                total_loss = sum(self.logs['loss']['train'])
                pbar.update(self.batch_size)
                pbar.set_postfix({'loss': total_loss / i + 1})

        end = time.time() - start

        self.logs['time'] = end