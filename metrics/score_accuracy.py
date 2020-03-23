import tensorflow as tf

class ScoreAccuracy(tf.keras.metrics.Metric):

    def __init__(self, y_max, name='score_accuracy', **kwargs):
        super(ScoreAccuracy, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.step = self.add_weight(name='step', initializer='zeros')
        self.all_gold = [i/(y_max-1) for i in range(y_max)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        print('Score Acc | y_pred type =', type(y_pred))
        y_pred = tf.map_fn(lambda y: min(self.all_gold, key=lambda x:abs(x-y)), y_pred)
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.sum.assign_add(tf.reduce_sum(values))
        self.step.assign_add(1.)

    def result(self):
        return self.sum / self.step

    def reset_states(self):
        self.sum.assign(0.)
        self.step.assign(0.)
