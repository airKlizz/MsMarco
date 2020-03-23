import tensorflow as tf
from transformers import TFAlbertModel, AlbertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model.scorer import Scorer

'''
Variables for dataset
'''
TRAIN_PATH = 'data/train/test.tsv'
MAX_LENGTH = 128
TEST_SIZE = 0.2
BATCH_SIZE = 16

'''
Variables for training
'''
NUMBER_EPOCHS = 1
LEARNING_RATE = 1e-5
EPSILON = 1e-8
CLIPNORM = 1.0

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
    y = [(i-min(y))/(max(y)-min(y)) for i in y]
    train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_X, validation_y)).batch(batch_size)
    return train_dataset, validation_dataset

def train_epoch(model, optimizer, loss, metrics, train_dataset, validation_dataset):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = model.fit(train_dataset, epochs=1, validation_data=validation_dataset)
    return history

def main():
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    print("CREATE MODEL")
    model = Scorer(tokenizer, TFAlbertModel, MAX_LENGTH)
    model.from_pretrained('albert-base-v2')
    print("CREATE DATASET")
    train_dataset, validation_dataset = create_tf_dataset(TRAIN_PATH, tokenizer, MAX_LENGTH, TEST_SIZE, BATCH_SIZE)
    print("TRAINING PREPARATION")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON, clipnorm=CLIPNORM)
    loss = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanSquaredError()]

    history = []
    for i in range(NUMBER_EPOCHS):
        print("EPOCH {}".format(i))
        history_epoch = train_epoch(model, optimizer, loss, metrics, train_dataset, validation_dataset)
        history.append(history_epoch)

    print('History:', history[0])


if __name__ == "__main__":
    main()