from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import os

class Item2Vec:
    def __init__(self, vector_size=200, alpha=0.035, epochs=20, window=15, workers=-1):
        if workers<0:
            workers=os.cpu_count()
        
        self.model = Word2Vec(
            vector_size=vector_size,
            alpha=alpha,
            epochs=epochs,
            window=window,
            workers=workers,
            min_count=0,
        )

        self.grouped_pdf = None

    def train(self, train_pdf, user_col='reviewerID', item_col='asin', permutations=True, corpus_file=None,
              update=False, epochs=None):
        if epochs is None:
            epochs = self.model.epochs

        self.grouped_pdf = train_pdf.groupby(user_col)[item_col].apply(
            lambda xs: list(map(str, xs))
        )

        temp_file = NamedTemporaryFile('w+')
        if corpus_file is None:
            for items in self.grouped_pdf.values:
                if permutations:
                    for _ in range(int(np.sqrt(len(items)))):
                        temp_file.write(' '.join(np.random.permutation(items)) + '\n')
                else:
                    temp_file.write(' '.join(items) + '\n')
            corpus_file = temp_file.name

        self.model.build_vocab(corpus_file=corpus_file, update=update)
        self.model.train(
            corpus_file=corpus_file,
            total_examples=self.model.corpus_count,
            epochs=epochs,
            total_words=self.model.corpus_total_words
        )

        temp_file.close()

    def calculate_user_embedding(self, items):
        items = list(map(self.model.wv.get_index, items))

        l1 = np.sum(self.model.wv.vectors[items], axis=0)
        if self.model.cbow_mean:
            l1 /= len(items)

        return l1

    def generate_item_embeddings(self):
        embeddings_pdf = pd.DataFrame(self.model.wv.vectors, index=np.array(self.model.wv.index_to_key, dtype=int))
        embeddings_pdf.columns = map(str, embeddings_pdf.columns)
        return embeddings_pdf.copy()

    def generate_user_embeddings(self):
        embeddings = []
        for items in self.grouped_pdf.values:
            embeddings.append(self.calculate_user_embedding(items))

        embeddings_pdf = pd.DataFrame(embeddings, index=self.grouped_pdf.index)
        embeddings_pdf.columns = map(str, embeddings_pdf.columns)
        return embeddings_pdf.copy()
