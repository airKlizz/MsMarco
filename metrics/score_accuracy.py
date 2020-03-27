import tensorflow as tf

class ScoreAccuracy(tf.keras.metrics.Metric):

    def __init__(self, y_max, name='score_accuracy', **kwargs):
        super(ScoreAccuracy, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.step = self.add_weight(name='step', initializer='zeros')
        self.interval = tf.constant(1 / (2 * (y_max-1)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(y_pred, (-1,))
        values = tf.math.abs(tf.cast(y_true, 'float32') - tf.cast(y_pred, 'float32')) <= self.interval
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.sum.assign_add(tf.reduce_sum(values))
        self.step.assign_add(tf.cast(tf.shape(y_true)[0], 'float32'))

    @classmethod
    def calculate_score_accuracy(cls, y_true, y_pred, interval):
        y_pred = tf.reshape(y_pred, (-1,))
        values = tf.math.abs(tf.cast(y_true, 'float32') - tf.cast(y_pred, 'float32')) <= interval
        values = tf.cast(values, 'float32')
        sum = tf.reduce_sum(values)
        step = tf.cast(tf.shape(y_true)[0], 'float32')
        return tf.math.divide(sum, step)

    def result(self):
        return tf.math.divide(self.sum, self.step)

    def reset_states(self):
        self.sum.assign(0.)
        self.step.assign(0.)
