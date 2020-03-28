import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from model.scorer import Scorer
from metrics.confusion_matrix import ConfusionMatrix
from mmr.run_mmr import EvaluationQueries
from mmr.msmarco_eval import compute_metrics_from_files

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_classes, shuffle=10000, random_state=2020):
    with open(train_path, 'r') as f:
        lines = f.readlines()

    X = []
    y = []
    class_num_samples = np.zeros(num_classes)
    for line in tqdm(lines, desc="Reading train file"):
        line = line.split('\t')
        assert len(line) == 5, print('\\t in querie or passage. \nQUERIE: {}\nPASSAGE: {}'.format(line[1], line[3]))
        X.append(tokenizer.encode(text=str(line[1]),
                                  text_pair=str(line[3]),
                                  max_length=max_length,
                                  pad_to_max_length=True))
        label = int(line[4][0])
        class_num_samples[label] += 1
        one_hot = [1 if i==label else 0 for i in range(num_classes)]
        y.append(one_hot)
    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    class_weight = np.sum(class_num_samples) / (np.shape(class_num_samples)[0] * class_num_samples)
    return train_dataset, validation_dataset, len(train_y)+1, len(validation_y)+1, class_weight

@tf.function
def train_step(model, optimizer, loss, inputs, gold, sample_weight, train_loss, train_acc, train_top_k_categorical_acc, train_confusion_matrix):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_value = loss(gold, predictions, sample_weight=sample_weight)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_value)
    train_acc(gold, predictions)
    train_top_k_categorical_acc(gold, predictions)
    train_confusion_matrix(gold, predictions)

@tf.function
def test_step(model, loss, inputs, gold, sample_weight, validation_loss, validation_acc, validation_top_k_categorical_acc, validation_confusion_matrix):
    predictions = model(inputs, training=False)
    t_loss = loss(gold, predictions, sample_weight=sample_weight)
    validation_loss(t_loss)
    validation_acc(gold, predictions)
    validation_top_k_categorical_acc(gold, predictions)
    validation_confusion_matrix(gold, predictions)

def main(model_name, train_path, max_length, test_size, batch_size, num_classes, epochs, learning_rate, epsilon, clipnorm, bm25_path, passages_path, queries_path, n_top, n_queries_to_evaluate, mrr_every, reference_path, candidate_path):
    '''
    Load Hugging Face tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Scorer(tokenizer, TFAutoModel, max_length, num_classes)
    model.from_pretrained(model_name)

    '''
    Create train and validation dataset
    '''
    train_dataset, validation_dataset, train_length, validation_length, class_weight = create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_classes)

    '''
    Initialize optimizer and loss function for training
    '''
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon, clipnorm=clipnorm)
    loss = tf.keras.losses.CategoricalCrossentropy()
    
    '''
    Define metrics
    '''
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    validation_acc = tf.keras.metrics.CategoricalAccuracy(name='validation_accuracy')
    train_top_k_categorical_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='train_top_2_categorical_accuracy')
    validation_top_k_categorical_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='validation_top_2_categorical_accuracy')
    train_confusion_matrix = ConfusionMatrix(num_classes, name='train_confusion_matrix')
    validation_confusion_matrix = ConfusionMatrix(num_classes, name='validation_confusion_matrix')

    mmr = EvaluationQueries(bm25_path, queries_path, passages_path, n_top)
    if n_queries_to_evaluate == -1:
        n_queries_to_evaluate = None

    '''
    Training loop over epochs
    '''
    for epoch in range(epochs):
        train_loss.reset_states()
        validation_loss.reset_states()
        train_acc.reset_states()
        validation_acc.reset_states()
        train_top_k_categorical_acc.reset_states()
        validation_top_k_categorical_acc.reset_states()
        train_confusion_matrix.reset_states()
        validation_confusion_matrix.reset_states()

        for inputs, gold in tqdm(train_dataset, desc="Training in progress", total=train_length/batch_size):
            labels = tf.argmax(gold, -1)
            sample_weight = class_weight[labels.numpy().astype(int)]
            train_step(model, optimizer, loss, inputs, gold, sample_weight, train_loss, train_acc, train_top_k_categorical_acc, train_confusion_matrix)

        for inputs, gold in tqdm(validation_dataset, desc="Validation in progress", total=validation_length/batch_size):
            labels = tf.argmax(gold, -1)
            sample_weight = class_weight[labels.numpy().astype(int)]
            test_step(model, loss, inputs, gold, class_weight, validation_loss, validation_acc, validation_top_k_categorical_acc, validation_confusion_matrix)

        template = '\nEpoch {}: \nTrain Loss: {}, Acc: {}, Top 2: {}, Confusion matrix:\n{}\nValidation Loss: {}, Acc: {}, Top 2: {}, Confusion matrix:\n{}'
        print(template.format(epoch+1,
                                train_loss.result(),
                                train_acc.result(),
                                train_top_k_categorical_acc.result(),
                                train_confusion_matrix.result(),
                                validation_loss.result(),
                                validation_acc.result(),
                                validation_top_k_categorical_acc.result().numpy,
                                validation_confusion_matrix.result()
                                ))
        if (epoch+1) % mrr_every == 0:
            mmr.score(model, candidate_path, n_queries_to_evaluate)
            mmr_metrics = compute_metrics_from_files(reference_path, candidate_path)
            print(
                'Queries ranked: {}, MRR @10: {}'.format(mmr_metrics['QueriesRanked'], mmr_metrics['MRR @10'])
            )


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
    parser.add_argument("--num_classes", type=int, help="number of output score class", default=4)
    
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
    main(args.model_name, args.train_path, args.max_length, args.test_size, args.batch_size, args.num_classes, args.epochs, args.learning_rate, args.epsilon, args.clipnorm, args.bm25_path, args.passages_path, args.queries_path, args.n_top, args.n_queries_to_evaluate, args.mrr_every, args.reference_path, args.candidate_path)