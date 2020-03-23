import tensorflow as tf

class ScoreAccuracy(tf.keras.metrics.Metric):

    def __init__(self, y_max, name='score_accuracy', **kwargs):
        super(ScoreAccuracy, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.step = self.add_weight(name='step', initializer='zeros')
        self.interval = tf.constant(1 / (2 * (y_max-1)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        print('\n\nSCORE ACCURACY | update state | y_true : {} | t_pred : {}\n\n'.format(y_true, y_pred))
        values = tf.math.abs(tf.cast(y_true, 'float32') - tf.cast(y_pred, 'float32')) <= self.interval
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.sum.assign_add(tf.reduce_sum(values))
        self.step.assign_add(1.)

    def result(self):
        return tf.math.divide(self.sum, self.step)

    def reset_states(self):
        self.sum.assign(0.)
        self.step.assign(0.)
