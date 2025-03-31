from src.data_generation import generate_synthetic_data
from src.train import train_model
from src.predict import load_model_and_scaler, predict_training_program

if __name__ == '__main__' :
    # Generate and save synthetic data (run once; comment out if already generated)
    data = generate_synthetic_data()
    data.to_csv('C:/Users/skard/PycharmProjects/deadlift_predictive/data/synthetic_data.csv', index=False)
    print(
        "Synthetic data generated and saved as 'C:/Users/skard/PycharmProjects/deadlift_predictive/data/synthetic_data.csv'.")


    # Train the extended model
    train_model()

    # Load model and scaler for prediction
    model, scaler = load_model_and_scaler()

    # Example input: [training_age, sets, reps, intensity, base_weight, tempo, training_frequency, motor_control]
    example_features = [2.5, 4, 8, 8.5, 150, 1.5, 3, 1]
    recommendation = predict_training_program(example_features, model, scaler)

    print("\nTraining Program Recommendation:")
    print(recommendation)
