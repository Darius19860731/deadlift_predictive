from .data_preprocessing import load_and_preprocess_data
from .model import build_extended_model


def train_model ( ) :
    (X_train, X_test,
     y_cat_train, y_cat_test,
     y_goal_train, y_goal_test,
     y_cycle_train, y_cycle_test,
     y_per_train, y_per_test,
     y_weekly_train, y_weekly_test,
     scaler) = load_and_preprocess_data()

    model = build_extended_model((X_train.shape[1],))

    history = model.fit(X_train,
                        {'category_output' : y_cat_train,
                         'goal_output' : y_goal_train,
                         'cycle_length_output' : y_cycle_train,
                         'periodization_output' : y_per_train,
                         'weekly_progression_output' : y_weekly_train},
                        epochs=50,
                        batch_size=16,
                        validation_split=0.2)

    results = model.evaluate(X_test,
                             {'category_output' : y_cat_test,
                              'goal_output' : y_goal_test,
                              'cycle_length_output' : y_cycle_test,
                              'periodization_output' : y_per_test,
                              'weekly_progression_output' : y_weekly_test})
    print("Test results:", results)

    # Save the extended model in the data folder
    model.save('C:/Users/skard/PycharmProjects/deadlift_predictive/data/deadlift_model_extended.h5')
    print(
        "Extended model saved as 'C:/Users/skard/PycharmProjects/deadlift_predictive/data/deadlift_model_extended.h5'.")


if __name__ == '__main__' :
    train_model()