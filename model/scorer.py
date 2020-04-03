import tensorflow as tf
import numpy as np

class Scorer(tf.keras.Model):
    def __init__(self, tokenizer, model, max_length, num_classes):
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
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(512, activation='sigmoid')
        self.classification = tf.keras.layers.Dense(num_classes, activation='softmax')
        self.max_length = max_length

    def from_pretrained(self, huggingface_model):
        self.model = self.model.from_pretrained(huggingface_model)

    def prepare_input(self, query, passage):
        inputs = self.tokenizer.encode_plus(text=query, 
                                     text_pair=passage,
                                     max_length=self.max_length,
                                     pad_to_max_length=True, 
                                     return_token_type_ids=True, 
                                     return_attention_mask=True)
        return [inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']]

    def prepare_inputs(self, queries, passages):
        inputs = []
        for query, passage in zip(queries, passages):
            inputs.append(self.prepare_input(query, passage))
        return inputs

    def call(self, inputs):
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        assert len(inputs) == 3, 'inputs: {}'.format(tf.shape(inputs))
        assert len(inputs[0]) == len(inputs[1]) and len(inputs[0]) == len(inputs[2])
        x = self.model(inputs[0], attention_mask=inputs[1], token_type_ids=inputs[2])
        x = self.flatten(x[0])
        x = self.dense(x)     
        x = self.classification(x)
        return x

    def score_query_passage(self, query, passage):
        return self.score_from_prediction(self.predict([self.prepare_input(query, passage)]))

    def score_query_passages(self, query, passages, batch_size):
        queries = [query] * len(passages)
        inputs = self.prepare_inputs(queries, passages)
        return self.score_from_prediction(self.predict(inputs, batch_size=batch_size))

    @staticmethod
    def score_from_prediction(prediction):
        #take value for the 4th class
        prediction = prediction[:, -1]
        np_prediction = np.asarray(prediction)
        return list(np_prediction)
        