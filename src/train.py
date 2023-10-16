from flair.models import SequenceTagger
from flair.data import Corpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, TransformerWordEmbeddings
from flair.trainers import ModelTrainer
from flair.datasets import ColumnCorpus
from datetime import datetime

#Retrieves date in string format to name training folder, avoids overwriting old data
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y__%H:%M:%S")

def get_tag_dictionary_and_corpus(data_folder):
    '''
    Get tag dictionary and corpus for data in data_folder.
    '''
    columns = {0: 'text', 1: 'ner'}
    data_folder = data_folder
    corpus: Corpus = ColumnCorpus(data_folder, columns, 
                                train_file='train.txt',
                                test_file='test.txt',
                                dev_file='dev.txt')

    print("Len train : ", len(corpus.train))
    print("Len test : ", len(corpus.test))
    print("Len dev : ", len(corpus.dev))
    print(corpus.train[0].to_tagged_string('ner'))

    tag_dictionary = corpus.make_label_dictionary(label_type="ner")
    print(tag_dictionary)
    return tag_dictionary,corpus

def train(tag_dictionary,corpus):
    """embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]"""
    embedding_types = [
        WordEmbeddings('fr'),
        FlairEmbeddings('fr-forward'),
        FlairEmbeddings('fr-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger = SequenceTagger(hidden_size=256,
                            rnn_layers=2,
                            embeddings=embeddings,
                            tag_dictionary=tag_dictionary,
                            tag_type="ner",
                            use_crf=True)
    
    trainer = ModelTrainer(tagger, corpus)

    trainer.train('resources/taggers/'+dt_string,
                learning_rate=0.2,
                mini_batch_size=12,
                max_epochs=180,
                embeddings_storage_mode='gpu',
                patience=6,
                anneal_factor=0.9)

data_folder = "train_data"    
tag_dictionary,corpus = get_tag_dictionary_and_corpus(data_folder)
train(tag_dictionary,corpus)