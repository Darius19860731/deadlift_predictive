import tensorflow as tf
from tensorflow.keras import layers, models, Input


def build_extended_model (input_shape) :
    inputs = Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)

    # Output 1: Category classification (5 classes)
    category_output = layers.Dense(5, activation='softmax', name='category_output')(x)

    # Output 2: Predicted end-cycle goal (regression)
    goal_output = layers.Dense(1, activation='linear', name='goal_output')(x)

    # Output 3: Recommended cycle length (regression)
    cycle_length_output = layers.Dense(1, activation='linear', name='cycle_length_output')(x)

    # Output 4: Periodization style (classification: 3 classes)
    periodization_output = layers.Dense(3, activation='softmax', name='periodization_output')(x)

    # Output 5: Weekly progression (regression)
    weekly_progression_output = layers.Dense(1, activation='linear', name='weekly_progression_output')(x)

    model = models.Model(inputs=inputs, outputs=[category_output, goal_output, cycle_length_output,
                                                 periodization_output, weekly_progression_output])

    model.compile(optimizer='adam',
                  loss={'category_output' : 'sparse_categorical_crossentropy',
                        'goal_output' : 'mean_squared_error',
                        'cycle_length_output' : 'mean_squared_error',
                        'periodization_output' : 'sparse_categorical_crossentropy',
                        'weekly_progression_output' : 'mean_squared_error'},
                  metrics={'category_output' : 'accuracy',
                           'goal_output' : 'mae',
                           'cycle_length_output' : 'mae',
                           'periodization_output' : 'accuracy',
                           'weekly_progression_output' : 'mae'})
    return model


if __name__ == '__main__' :
    model = build_extended_model((8,))
    model.summary()
