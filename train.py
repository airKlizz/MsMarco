import tensorflow as tf
from transformers import TFAlbertModel, AlbertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.scorer import Scorer
from metrics.score_accuracy import ScoreAccuracy
from mmr.run_mmr import EvaluationQueries
from mmr.msmarco_eval import compute_metrics_from_files

'''
Variables for dataset
'''
TRAIN_PATH = 'data/train/test.tsv'
MAX_LENGTH = 128
TEST_SIZE = 0.2
BATCH_SIZE = 8

'''
Variables for training
'''
EPOCHS = 3
LEARNING_RATE = 5e-6
EPSILON = 1e-8
CLIPNORM = 1.0

'''
Variables for eval
'''
BM25_PATH = "data/evaluation/bm25/run.dev.small.tsv"
PASSAGES_PATH = "data/passages/passages.bm25.small.json"
QUERIES_PATH = "data/queries/queries.dev.small.tsv"
N_TOP = 3
N_QUERIES_TO_EVALUATE = 100 #None if all
REFERENCE_PATH = 'data/evaluation/gold/qrels.dev.small.tsv'
CANDIDATE_PATH = 'data/evaluation/albert-base-v2/run.tsv'

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

def main():
    '''
    Load Hugging Face tokenizer and model
    '''
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model = Scorer(tokenizer, TFAlbertModel, MAX_LENGTH)
    model.from_pretrained('albert-base-v2')

    '''
    Create train and validation dataset
    '''
    train_dataset, validation_dataset, train_length, validation_length, y_max = create_tf_dataset(TRAIN_PATH, tokenizer, MAX_LENGTH, TEST_SIZE, BATCH_SIZE)
    ## Reduce dataset to run on local machine 
    ## Need to be removed for the real training
    train_dataset = train_dataset.take(10)
    validation_dataset = validation_dataset.take(10)
    train_length = 10 * BATCH_SIZE
    validation_length = 10 * BATCH_SIZE
    ## End


    '''
    Initialize optimizer and loss function for training
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON, clipnorm=CLIPNORM)
    loss = tf.keras.losses.MeanSquaredError()
    
    '''
    Define metrics
    '''
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    train_acc = ScoreAccuracy(y_max, name='train_score_accuracy')
    validation_acc = ScoreAccuracy(y_max, name='validation_score_accuracy')
    mmr = EvaluationQueries(BM25_PATH, QUERIES_PATH, PASSAGES_PATH, N_TOP)

    '''
    Training loop over epochs
    '''
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        validation_loss.reset_states()

        for inputs, gold in tqdm(train_dataset, desc="Training in progress", total=train_length/BATCH_SIZE):
            train_step(model, optimizer, loss, inputs, gold, train_loss, train_acc)

        for inputs, gold in tqdm(validation_dataset, desc="Validation in progress", total=validation_length/BATCH_SIZE):
            test_step(model, loss, inputs, gold, validation_loss, validation_acc)

        mmr.score(model, CANDIDATE_PATH, N_QUERIES_TO_EVALUATE)
        mmr_metrics = compute_metrics_from_files(REFERENCE_PATH, CANDIDATE_PATH)
        
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
    main()