import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from model.scorer import Scorer
from metrics.confusion_matrix import ConfusionMatrix
from mrr.run_mrr import EvaluationQueries
from mrr.msmarco_eval import compute_metrics_from_files

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_samples, shuffle=10000, random_state=2020):
    X = []
    y = []
    count = 0
    with open(train_path, 'r') as f:
        for line in tqdm(f, desc="Reading train file"):
            count += 1
            if count >= num_samples:
                break
            line = line.split('\t')
            assert len(line) == 3, '\\t in querie or passage. \nQUERIE: {}\nPASSAGE1: {}\nPASSAGE2: {}'.format(line[0], line[1], line[2])
            # Add relevant passage
            relevant_inputs = tokenizer.encode_plus(text=str(line[0]),
                                    text_pair=str(line[1]),
                                    max_length=max_length,
                                    pad_to_max_length=True,
                                    return_token_type_ids=True, 
                                    return_attention_mask=True)
            X.append([relevant_inputs['input_ids'],
                      relevant_inputs['attention_mask'],
                      relevant_inputs['token_type_ids']                   
                     ])
            y.append([0, 1])
            # Add no relevant passage
            no_relevant_inputs = tokenizer.encode_plus(text=str(line[0]),
                                    text_pair=str(line[2]),
                                    max_length=max_length,
                                    pad_to_max_length=True, 
                                    return_token_type_ids=True, 
                                    return_attention_mask=True)
            X.append([no_relevant_inputs['input_ids'],
                      no_relevant_inputs['attention_mask'],
                      no_relevant_inputs['token_type_ids']                   
                     ])
            y.append([1, 0])
    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    return train_dataset, validation_dataset, len(train_y)+1, len(validation_y)+1

@tf.function
def train_step(model, optimizer, loss, inputs, gold, train_loss, train_acc, train_top_k_categorical_acc, train_confusion_matrix):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_value = loss(gold, predictions)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss_value)
    train_acc(gold, predictions)
    train_top_k_categorical_acc(gold, predictions)
    train_confusion_matrix(gold, predictions)

@tf.function
def test_step(model, loss, inputs, gold, validation_loss, validation_acc, validation_top_k_categorical_acc, validation_confusion_matrix):
    predictions = model(inputs, training=False)
    t_loss = loss(gold, predictions)
    validation_loss(t_loss)
    validation_acc(gold, predictions)
    validation_top_k_categorical_acc(gold, predictions)
    validation_confusion_matrix(gold, predictions)

def main(model_name, train_path, max_length, test_size, batch_size, num_samples, num_classes, epochs, learning_rate, epsilon, clipnorm, save_path, bm25_path, passages_path, queries_path, n_top, n_queries_to_evaluate, mrr_every, reference_path, candidate_path):
    '''
    Load Hugging Face tokenizer and model
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Scorer(tokenizer, TFAutoModel, max_length, num_classes)
    model.from_pretrained(model_name)

    '''
    Create train and validation dataset
    '''
    train_dataset, validation_dataset, train_length, validation_length = create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_samples)

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

    mrr = EvaluationQueries(bm25_path, queries_path, passages_path, n_top)
    if n_queries_to_evaluate == -1:
        n_queries_to_evaluate = None

    '''
    Training loop over epochs
    '''
    model_save_path_template = save_path+'model_{model_name}_epoch_{epoch:04d}_mrr_{mrr:.3f}.h5'
    model_save_path_step_template = save_path+'model_{model_name}_epoch_{epoch:04d}_step_{step:04d}_loss_{loss:.3f}.h5'
    template_step = '\nStep {}: \nTrain Loss: {}, Acc: {}, Top 2: {}, Confusion matrix:\n{}\nValidation Loss: {}, Acc: {}, Top 2: {}, Confusion matrix:\n{}'
    template_epoch = '\nEpoch {}: \nTrain Loss: {}, Acc: {}, Top 2: {}, Confusion matrix:\n{}\nValidation Loss: {}, Acc: {}, Top 2: {}, Confusion matrix:\n{}'
    previus_mrr = 0.19
    previus_validation_loss = 10000000

    for epoch in range(epochs):
        train_loss.reset_states()
        validation_loss.reset_states()
        train_acc.reset_states()
        validation_acc.reset_states()
        train_top_k_categorical_acc.reset_states()
        validation_top_k_categorical_acc.reset_states()
        train_confusion_matrix.reset_states()
        validation_confusion_matrix.reset_states()

        training_step = 0
        for inputs, gold in tqdm(train_dataset, desc="Training in progress", total=int(train_length/batch_size+1)):
            training_step += 1
            train_step(model, optimizer, loss, inputs, gold, train_loss, train_acc, train_top_k_categorical_acc, train_confusion_matrix)
            
            '''
            Validation loop every XXXX steps
            '''
            if (training_step-1) % 2000 == 0:  

                for inputs, gold in tqdm(validation_dataset, desc="Validation in progress", total=int(validation_length/batch_size+1)):
                    test_step(model, loss, inputs, gold, validation_loss, validation_acc, validation_top_k_categorical_acc, validation_confusion_matrix)

                print(template_step.format(training_step+1,
                                train_loss.result(),
                                train_acc.result(),
                                train_top_k_categorical_acc.result(),
                                train_confusion_matrix.result(),
                                validation_loss.result(),
                                validation_acc.result(),
                                validation_top_k_categorical_acc.result(),
                                validation_confusion_matrix.result()
                                ))

                if previus_validation_loss > validation_loss.result().numpy():
                    previus_validation_loss = validation_loss.result().numpy()
                    model_save_path_step = model_save_path_step_template.format(model_name=model_name, epoch=epoch, step=training_step, loss=previus_validation_loss)
                    print('Saving: ', model_save_path_step)
                    model.save_weights(model_save_path_step, save_format='h5')

                train_loss.reset_states()
                validation_loss.reset_states()
                train_acc.reset_states()
                validation_acc.reset_states()
                train_top_k_categorical_acc.reset_states()
                validation_top_k_categorical_acc.reset_states()
                train_confusion_matrix.reset_states()
                validation_confusion_matrix.reset_states()

        for inputs, gold in tqdm(validation_dataset, desc="Validation in progress", total=int(validation_length/batch_size+1)):
            test_step(model, loss, inputs, gold, validation_loss, validation_acc, validation_top_k_categorical_acc, validation_confusion_matrix)
        
        print(template_epoch.format(epoch+1,
                                train_loss.result(),
                                train_acc.result(),
                                train_top_k_categorical_acc.result(),
                                train_confusion_matrix.result(),
                                validation_loss.result(),
                                validation_acc.result(),
                                validation_top_k_categorical_acc.result(),
                                validation_confusion_matrix.result()
                                ))
        if (epoch+1) % mrr_every == 0:
            mrr.score(model, candidate_path, n_queries_to_evaluate)
            mrr_metrics = compute_metrics_from_files(reference_path, candidate_path)
            print(
                'Queries ranked: {}, MRR @10: {}'.format(mrr_metrics['QueriesRanked'], mrr_metrics['MRR @10'])
            )
            if mrr_metrics['MRR @10'] > previus_mrr:
                previus_mrr = mrr_metrics['MRR @10']
                model_save_path = model_save_path_template.format(model_name=model_name, epoch=epoch, mrr=previus_mrr)
                print('Saving: ', model_save_path)
                model.save_weights(model_save_path, save_format='h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for the model
    '''
    parser.add_argument("--model_name", type=str, help="Name of the HugginFace Model", default="bert-base-uncased")

    '''
    Variables for dataset
    '''
    parser.add_argument("--train_path", type=str, help="path to the train .tsv file", default="data/train/triples.train.small.tsv")
    parser.add_argument("--max_length", type=int, help="max length of the tokenized input", default=256)
    parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
    parser.add_argument("--batch_size", type=int, help="batch size", default=12)
    parser.add_argument("--num_classes", type=int, help="number of output score class", default=2)
    parser.add_argument("--num_samples", type=int, help="number of samples", default=50000)
    
    '''
    Variables for training
    '''
    parser.add_argument("--epochs", type=int, help="number of epochs", default=5)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--epsilon", type=float, help="epsilon", default=1e-8)
    parser.add_argument("--clipnorm", type=float, help="clipnorm", default=1.0)
    parser.add_argument("--save_path", type=str, help="path to the save folder", default="model/saved_weights/")

    '''
    Variables for evaluation
    '''
    parser.add_argument("--bm25_path", type=str, help="path to the BM25 run .tsv file", default="data/evaluation/bm25/run.dev.small.tsv")
    parser.add_argument("--passages_path", type=str, help="path to the BM25 passages .json file", default="data/passages/passages.bm25.small.json")
    parser.add_argument("--queries_path", type=str, help="path to the BM25 queries .tsv file", default="data/queries/queries.dev.small.tsv")
    parser.add_argument("--n_top", type=int, help="number of passages to re-rank after BM25", default=50)
    parser.add_argument("--n_queries_to_evaluate", type=int, help="number of queries to evaluate for MMR", default=1000)
    parser.add_argument("--mrr_every", type=int, help="number of epochs between mrr eval", default=1)
    parser.add_argument("--reference_path", type=str, help="path to the reference gold .tsv file", default="data/evaluation/gold/qrels.dev.small.tsv")
    parser.add_argument("--candidate_path", type=str, help="path to the candidate run .tsv file", default="data/evaluation/model/run.tsv")
    
    '''
    Run main
    '''
    args = parser.parse_args()
    main(args.model_name, args.train_path, args.max_length, args.test_size, args.batch_size, args.num_samples, args.num_classes, args.epochs, args.learning_rate, args.epsilon, args.clipnorm, args.save_path, args.bm25_path, args.passages_path, args.queries_path, args.n_top, args.n_queries_to_evaluate, args.mrr_every, args.reference_path, args.candidate_path)