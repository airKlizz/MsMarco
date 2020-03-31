import tensorflow as tf
import numpy as np
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
from model.scorer import Scorer
from metrics.confusion_matrix import ConfusionMatrix
from squad.squad_dataset import SquadDataset

def create_tf_dataset(train_path, tokenizer, max_length, test_size, batch_size, num_samples, shuffle=10000, random_state=2020):
    squad_dataset = SquadDataset(train_path)
    train_dataset, validation_dataset, train_length, validation_length = squad_dataset.return_tensorflow_tokenized_dataset(tokenizer, max_length, test_size, batch_size, num_samples, shuffle, random_state)
    del squad_dataset
    return train_dataset, validation_dataset, train_length, validation_length


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

def main(model_name, train_path, max_length, test_size, batch_size, num_samples, num_classes, epochs, learning_rate, epsilon, clipnorm):
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
            train_step(model, optimizer, loss, inputs, gold, train_loss, train_acc, train_top_k_categorical_acc, train_confusion_matrix)

        for inputs, gold in tqdm(validation_dataset, desc="Validation in progress", total=validation_length/batch_size):
            test_step(model, loss, inputs, gold, validation_loss, validation_acc, validation_top_k_categorical_acc, validation_confusion_matrix)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    Variables for the model
    '''
    parser.add_argument("--model_name", type=str, help="Name of the HugginFace Model", default="bert-base-uncased")

    '''
    Variables for dataset
    '''
    parser.add_argument("--train_path", type=str, help="path to the train .tsv file", default="squad/data/train-v2.0.json")
    parser.add_argument("--max_length", type=int, help="max length of the tokenized input", default=256)
    parser.add_argument("--test_size", type=float, help="ratio of the test dataset", default=0.2)
    parser.add_argument("--batch_size", type=int, help="batch size", default=12)
    parser.add_argument("--num_classes", type=int, help="number of output score class", default=3)
    parser.add_argument("--num_samples", type=int, help="number of samples (None means all samples)", default=None)
    
    '''
    Variables for training
    '''
    parser.add_argument("--epochs", type=int, help="number of epochs", default=20)
    parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--epsilon", type=float, help="epsilon", default=1e-8)
    parser.add_argument("--clipnorm", type=float, help="clipnorm", default=1.0)

    '''
    Run main
    '''
    args = parser.parse_args()
    main(args.model_name, args.train_path, args.max_length, args.test_size, args.batch_size, args.num_samples, args.num_classes, args.epochs, args.learning_rate, args.epsilon, args.clipnorm)