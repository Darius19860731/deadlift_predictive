import numpy as np
import tensorflow as tf
from data_preprocessing import load_and_preprocess_data


def load_model_and_scaler (
        model_path='C:/Users/skard/PycharmProjects/deadlift_predictive/data/deadlift_model_extended.h5') :
    model = tf.keras.models.load_model(model_path)
    # Rest of your function...

    # Reload the scaler by running data preprocessing
    _, _, _, _, _, _, _, _, _, _, _, _, scaler = load_and_preprocess_data()
    return model, scaler


def predict_training_program (input_features, model, scaler) :
    """
    Expects input_features as a list of 8 elements:
    [training_age, sets, reps, intensity, base_weight, tempo, training_frequency, motor_control]
    """
    input_scaled = scaler.transform(np.array([input_features]))

    # Model prediction returns 5 outputs
    pred_category_prob, pred_goal, pred_cycle_length, pred_periodization_prob, pred_weekly_prog = model.predict(
        input_scaled)

    pred_category = int(np.argmax(pred_category_prob, axis=1)[0])
    pred_goal = pred_goal[0][0]
    pred_cycle_length = pred_cycle_length[0][0]
    pred_periodization = int(np.argmax(pred_periodization_prob, axis=1)[0])
    weekly_progression = pred_weekly_prog[0][0]

    periodization_labels = {0 : "linear", 1 : "DUP", 2 : "block"}

    rec_text = (
            f"Predicted Category: {pred_category} " +
            f"(e.g., {'Novice' if pred_category == 0 else 'Intermediate' if pred_category == 1 else 'Advanced' if pred_category == 2 else 'Master' if pred_category == 3 else 'Grand Master'}).\n" +
            f"Predicted End-Cycle Goal: {pred_goal:.2f} kg.\n" +
            f"Recommended Cycle Length: {pred_cycle_length:.1f} weeks.\n" +
            f"Suggested Periodization: {periodization_labels.get(pred_periodization, 'Unknown')}.\n" +
            f"Predicted Weekly Progression: {weekly_progression:.2f} kg/week."
    )
    return rec_text


if __name__ == '__main__' :
    model, scaler = load_model_and_scaler()
    example_features = [2.5, 4, 8, 8.5, 150, 1.5, 3, 1]
    recommendation = predict_training_program(example_features, model, scaler)
    print("Training Program Recommendation:")
    print(recommendation)
