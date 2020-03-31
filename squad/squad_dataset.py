import json
import tensorflow as tf
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class SquadDataset:
  def __init__(self, train_path):

    with open(train_path) as data_file:
      self.squad_data = json.load(data_file)['data']

  def create_data(self):

    contexts = []
    questions = []
    for _, article in enumerate(self.squad_data):

      article_contexts = []
      article_questions = []
      for _, paragraph in enumerate(article['paragraphs']):
        context = paragraph['context']
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')

        paragraph_contexts = []
        paragraph_questions = []
        for question in paragraph['qas']:
          if question['is_impossible'] == True:
            continue
          q = question['question']
          paragraph_contexts.append(context)
          paragraph_questions.append(q)

        article_contexts.append(paragraph_contexts)
        article_questions.append(paragraph_questions)
      
      contexts.append(article_contexts)
      questions.append(article_questions)

    self.data = {'contexts': contexts, 'questions': questions}

  def create_linked_data(self):

    self.linked_data = []
    for i in range(len(self.data['contexts'])):
      for j in range(len(self.data['contexts'][i])):
        for k in range(len(self.data['contexts'][i][j])):
          self.linked_data.append({
                'context': self.data['contexts'][i][j][k],
                'question': self.data['questions'][i][j][k]})
          
  def create_related_data(self):

    self.related_data = []
    for i in range(len(self.data['contexts'])):
      for j in range(len(self.data['contexts'][i])):
        for k in range(len(self.data['contexts'][i][j])):
          jbis = j
          while jbis == j:
            jbis = randint(0, len(self.data['contexts'][i])-1)
            if len(self.data['contexts'][i][jbis]) == 0:
              jbis = j
              continue
            kbis = randint(0, len(self.data['contexts'][i][jbis])-1)
          self.related_data.append({
                'context': self.data['contexts'][i][j][k],
                'question': self.data['questions'][i][jbis][kbis]})
          
  def create_unrelated_data(self):

    self.unrelated_data = []
    for i in range(len(self.data['contexts'])):
      for j in range(len(self.data['contexts'][i])):
        for k in range(len(self.data['contexts'][i][j])):
          ibis = i
          while ibis == i:
            ibis = randint(0, len(self.data['contexts'])-1)
            jbis = randint(0, len(self.data['contexts'][ibis])-1)
            if len(self.data['contexts'][ibis][jbis]) == 0:
              ibis = i
              continue
            kbis = randint(0, len(self.data['contexts'][ibis][jbis])-1)
          self.unrelated_data.append({
                'context': self.data['contexts'][i][j][k],
                'question': self.data['questions'][ibis][jbis][kbis]})
          
  def create_dataset(self):

    self.create_data()
    self.create_linked_data()
    self.create_related_data()
    self.create_unrelated_data()

    self.X = self.linked_data + self.related_data + self.unrelated_data
    self.y = [[0, 0, 1] for i in range(len(self.linked_data))] + [[0, 1, 0] for i in range(len(self.related_data))] + [[1, 0, 0] for i in range(len(self.unrelated_data))]
    self.X, _, self.y, _ = train_test_split(self.X, self.y, random_state=2020, test_size=1)
    print('Dataset Created')

  def create_tokenized_dataset(self, tokenizer, max_length, num_samples):
    
    self.create_dataset()
    self.X = self.X[:num_samples]
    self.y = self.y[:num_samples]

    
    self.X_tokenized = []
    for elem in tqdm(self.X, desc='Tokenize in progress'):
      inputs = tokenizer.encode_plus(text=elem['question'], text_pair=elem['context'], max_length=256, pad_to_max_length=True)
      self.X_tokenized.append([inputs['input_ids'],
                inputs['attention_mask'],
                inputs['token_type_ids']                   
                ])
    print('Tokenized Dataset Created')

  def return_tensorflow_tokenized_dataset(self, tokenizer, max_length, test_size, batch_size, num_samples, shuffle=10000, random_state=2020):
    self.create_tokenized_dataset(tokenizer, max_length, num_samples)
    self.train_X, self.validation_X, self.train_y, self.validation_y = train_test_split(np.array(self.X_tokenized), np.array(self.y), random_state=random_state, test_size=test_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((self.train_X, self.train_y)).shuffle(shuffle).batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((self.validation_X, self.validation_y)).batch(batch_size)
    return train_dataset, validation_dataset, int(len(self.train_y)/batch_size+1), int(len(self.validation_y)/batch_size+1)