import tensorflow as tf
import numpy as np

class Scorer(tf.keras.Model):
    def __init__(self, tokenizer, model, max_length):
        '''
        Scorer is a HuggingFace model with a scorer head to perform a passage scoring based on a query.

        Arguments:
            - tokenizer: HuggingFace tokenizer
            - model: HuggingFace model
            - max_length: max number of tokens for the input
        '''
        super(Scorer, self).__init__(name='Scorer')
        self.tokenizer = tokenizer
        self.model = model
        self.dense = tf.keras.layers.Dense(512, activation='relu')
        self.score = tf.keras.layers.Dense(1, activation='sigmoid')
        self.max_length = max_length

    def from_pretrained(self, huggingface_model):
        self.model = self.model.from_pretrained(huggingface_model)

    def prepare_input(self, query, passage):
        return self.tokenizer.encode(text=query, 
                                     text_pair=passage,
                                     max_length=self.max_length,
                                     pad_to_max_length=True)

    def prepare_inputs(self, queries, passages):
        inputs = []
        for query, passage in zip(queries, passages):
            inputs.append(self.prepare_input(query, passage))
        return inputs

    def call(self, inputs):
        x = self.model(inputs)[1] # x.shape = (None, 768)
        x = self.dense(x)     
        x = tf.reshape(self.score(x), (-1,))
        assert tf.shape(x)[0] == tf.shape(inputs)[0], 'Error in the output shape : {}'.format(x)
        return x

    def score_query_passage(self, query, passage):
        return list(np.asarray(tf.reshape(self.predict([self.prepare_input(query, passage)]), (-1,))))

    def score_query_passages(self, query, passages, batch_size):
        queries = [query] * len(passages)
        inputs = self.prepare_inputs(queries, passages)
        return list(np.asarray(tf.reshape(self.predict(inputs, batch_size=batch_size), (-1,))))