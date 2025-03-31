import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data ( ) :
    file_path = 'C:/Users/skard/PycharmProjects/deadlift_predictive/data/synthetic_data.csv'
    data = pd.read_csv(file_path)

    feature_cols = ['training_age', 'sets', 'reps', 'intensity', 'base_weight', 'tempo', 'training_frequency',
                    'motor_control']
    X = data[feature_cols]

    # Define target variables
    y_category = data['category']
    y_goal = data['predicted_goal']
    y_cycle_length = data['cycle_length']
    y_periodization = data['periodization']
    y_weekly = data['weekly_progression']

    # Split into training and test sets (80/20 split)
    splits = train_test_split(X, y_category, y_goal, y_cycle_length, y_periodization, y_weekly,
                              test_size=0.2, random_state=42)
    X_train, X_test, y_cat_train, y_cat_test, y_goal_train, y_goal_test, y_cycle_train, y_cycle_test, y_per_train, y_per_test, y_weekly_train, y_weekly_test = splits

    # Normalize the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_cat_train.values, y_cat_test.values, y_goal_train.values, y_goal_test.values, y_cycle_train.values, y_cycle_test.values, y_per_train.values, y_per_test.values, y_weekly_train.values, y_weekly_test.values, scaler


if __name__ == '__main__' :
    results = load_and_preprocess_data()
    print("Data loaded and preprocessed.")