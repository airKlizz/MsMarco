import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
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
        loss_value = loss(gold, predictions)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_value)
    train_acc(gold, predictions)
    return predictions, loss_value

@tf.function
def test_step(model, loss, inputs, gold, validation_loss, validation_acc):
    predictions = model(inputs, training=False)
    t_loss = loss(gold, predictions)
    validation_loss(t_loss)
    validation_acc(gold, predictions)
    return predictions, t_loss

def main(model_name, train_path, max_length, test_size, batch_size, epochs, learning_rate, epsilon, clipnorm, bm25_path, passages_path, queries_path, n_top, n_queries_to_evaluate, mrr_every, reference_path, candidate_path):
    '''
    Load Hugging Face tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Scorer(tokenizer, TFAutoModel, max_length)
    model.from_pretrained(model_name)

    '''
    Create train and validation dataset
    '''
    train_dataset, validation_dataset, _, _, _ = create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size)

    '''
    Initialize optimizer and loss function for training
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)
    #loss = tf.keras.losses.MeanSquareError()
    
    '''
    Define metrics
    '''
    model.compile(
        optimizer=optimizer, loss='mse', metrics=['mae', 'mse'],
    )
    
    '''
    Training loop over epochs
    '''
    model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for the model
    '''
    parser.add_argument("--model_name", type=str, help="Name of the HugginFace Model", default="bert-large-uncased")

    '''
    Variables for dataset
    '''
    parser.add_argument("--train_path", type=str, help="path to the train .tsv file", default="data/train/test.tsv")
    parser.add_argument("--max_length", type=int, help="max length of the tokenized input", default=256)
    parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
    parser.add_argument("--batch_size", type=int, help="batch size", default=24)
    
    '''
    Variables for training
    '''
    parser.add_argument("--epochs", type=int, help="number of epochs", default=20)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=5e-5)
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
    parser.add_argument("--mrr_every", type=int, help="number of epochs between mrr eval", default=5)
    parser.add_argument("--reference_path", type=str, help="path to the reference gold .tsv file", default="data/evaluation/gold/qrels.dev.small.tsv")
    parser.add_argument("--candidate_path", type=str, help="path to the candidate run .tsv file", default="data/evaluation/model/run.tsv")
    
    '''
    Run main
    '''
    args = parser.parse_args()
    main(args.model_name, args.train_path, args.max_length, args.test_size, args.batch_size, args.epochs, args.learning_rate, args.epsilon, args.clipnorm, args.bm25_path, args.passages_path, args.queries_path, args.n_top, args.n_queries_to_evaluate, args.mrr_every, args.reference_path, args.candidate_path)