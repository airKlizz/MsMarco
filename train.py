import tensorflow as tf
from transformers import TFAlbertModel, AlbertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from model.scorer import Scorer
from metrics.score_accuracy import ScoreAccuracy
from mmr.run_mmr import EvaluationQueries
from mmr.msmarco_eval import compute_metrics_from_files

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, shuffle=10000, random_state=2020):
    with open(train_path, 'r') as f:
        lines = f.readlines()

    X = []
    y = []
    for line in tqdm(lines, desc="Reading train file"):
        line = line.split('\t')
        assert len(line) == 5, print('\\t in querie or passage. \nQUERIE: {}\nPASSAGE: {}'.format(line[1], line[3]))
        X.append(tokenizer.encode(text=str(line[1]),
                                  text_pair=str(line[3]),
                                  max_length=max_length,
                                  pad_to_max_length=True))
        y.append(int(line[4][0]))
    y_max = max(y)
    y = [(i-min(y))/(max(y)-min(y)) for i in y]
    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    return train_dataset, validation_dataset, len(train_y)+1, len(validation_y)+1, y_max

@tf.function
def train_step(model, optimizer, loss, inputs, gold, train_loss, train_acc):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss(gold, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_acc(gold, predictions)

@tf.function
def test_step(model, loss, inputs, gold, validation_loss, validation_acc):
    predictions = model(inputs, training=False)
    t_loss = loss(gold, predictions)
    validation_loss(t_loss)
    validation_acc(gold, predictions)

def main(train_path, max_length, test_size, batch_size, epochs, learning_rate, epsilon, clipnorm, bm25_path, passages_path, queries_path, n_top, n_queries_to_evaluate, reference_path, candidate_path):
    '''
    Load Hugging Face tokenizer and model
    '''
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model = Scorer(tokenizer, TFAlbertModel, max_length)
    model.from_pretrained('albert-base-v2')

    '''
    Create train and validation dataset
    '''
    train_dataset, validation_dataset, train_length, validation_length, y_max = create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size)
    '''## Reduce dataset to run on local machine 
    ## Need to be removed for the real training
    train_dataset = train_dataset.take(10)
    validation_dataset = validation_dataset.take(10)
    train_length = 10 * batch_size
    validation_length = 10 * batch_size
    ## End'''


    '''
    Initialize optimizer and loss function for training
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)
    loss = tf.keras.losses.MeanSquaredError()
    
    '''
    Define metrics
    '''
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    train_acc = ScoreAccuracy(y_max, name='train_score_accuracy')
    validation_acc = ScoreAccuracy(y_max, name='validation_score_accuracy')
    mmr = EvaluationQueries(bm25_path, queries_path, passages_path, n_top)
    if n_queries_to_evaluate == -1:
        n_queries_to_evaluate = None

    '''
    Training loop over epochs
    '''
    for epoch in range(epochs):
        train_loss.reset_states()
        validation_loss.reset_states()

        for inputs, gold in tqdm(train_dataset, desc="Training in progress", total=train_length/batch_size):
            train_step(model, optimizer, loss, inputs, gold, train_loss, train_acc)

        for inputs, gold in tqdm(validation_dataset, desc="Validation in progress", total=validation_length/batch_size):
            test_step(model, loss, inputs, gold, validation_loss, validation_acc)

        mmr.score(model, candidate_path, n_queries_to_evaluate)
        mmr_metrics = compute_metrics_from_files(reference_path, candidate_path)
        
        template = 'Epoch {}, Loss: {}, Acc: {}, Validation Loss: {}, Validation Acc: {}, Queries ranked: {}, MRR @10: {}'
        print(template.format(epoch+1,
                                train_loss.result(),
                                train_acc.result(),
                                validation_loss.result(),
                                validation_acc.result(),
                                mmr_metrics['QueriesRanked'],
                                mmr_metrics['MRR @10']
                                ))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for dataset
    '''
    parser.add_argument("--train_path", type=str, help="path to the train .tsv file", default="data/train/test.tsv")
    parser.add_argument("--max_length", type=int, help="max length of the tokenized input", default=128)
    parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    
    '''
    Variables for training
    '''
    parser.add_argument("--epochs", type=int, help="number of epochs", default=3)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=5e-6)
    parser.add_argument("--epsilon", type=float, help="epsilon", default=1e-8)
    parser.add_argument("--clipnorm", type=float, help="clipnorm", default=1.0)

    '''
    Variables for evaluation
    '''
    parser.add_argument("--bm25_path", type=str, help="path to the BM25 run .tsv file", default="data/evaluation/bm25/run.dev.small.tsv")
    parser.add_argument("--passages_path", type=str, help="path to the BM25 passages .json file", default="data/passages/passages.bm25.small.json")
    parser.add_argument("--queries_path", type=str, help="path to the BM25 queries .tsv file", default="data/queries/queries.dev.small.tsv")
    parser.add_argument("--n_top", type=int, help="number of passages to re-rank after BM25", default=50)
    parser.add_argument("--n_queries_to_evaluate", type=int, help="number of queries to evaluate for MMR", default=-1)
    parser.add_argument("--reference_path", type=str, help="path to the reference gold .tsv file", default="data/evaluation/gold/qrels.dev.small.tsv")
    parser.add_argument("--candidate_path", type=str, help="path to the candidate run .tsv file", default="data/evaluation/albert-base-v2/run.tsv")
    
    '''
    Run main
    '''
    args = parser.parse_args()
    main(args.train_path, args.max_length, args.test_size, args.batch_size, args.epochs, args.learning_rate, args.epsilon, args.clipnorm, args.bm25_path, args.passages_path, args.queries_path, args.n_top, args.n_queries_to_evaluate, args.reference_path, args.candidate_path)