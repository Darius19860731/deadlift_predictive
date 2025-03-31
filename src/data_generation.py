import numpy as np
import pandas as pd


def assign_category (training_age) :
    # Categories based on training age (in years)
    if training_age < 1.5 :
        return 0  # Novice
    elif training_age < 3 :
        return 1  # Intermediate
    elif training_age < 5 :
        return 2  # Advanced
    elif training_age < 7 :
        return 3  # Master
    else :
        return 4  # Grand Master


# Mappings for cycle length (in weeks) and periodization style per category
cycle_length_mapping = {0 : 8, 1 : 10, 2 : 12, 3 : 14, 4 : 16}
periodization_mapping = {0 : 0, 1 : 0, 2 : 1, 3 : 2, 4 : 0}  # 0: linear, 1: DUP, 2: block


def generate_synthetic_data (num_samples=500) :
    np.random.seed(42)

    # Generate synthetic input features
    training_age = np.random.uniform(0.5, 10, num_samples)  # in years
    sets = np.random.randint(3, 6, num_samples)  # 3-5 sets
    reps = np.random.randint(3, 10, num_samples)  # 3-9 reps
    intensity = np.round(np.random.uniform(7.0, 10.0, num_samples), 1)  # intensity on a 7-10 scale
    base_weight = np.random.randint(100, 200, num_samples)  # base deadlift weight (kg)
    tempo = np.round(np.random.uniform(2, 5, num_samples), 1)  # seconds per rep (time under tension)
    training_frequency = np.random.randint(2, 5, num_samples)  # sessions per week
    motor_control = np.random.uniform(1, 10, num_samples)  # motor control rating (1-10)

    # Compute outputs
    categories = np.array([assign_category(age) for age in training_age])

    # Synthetic formula for predicted end-cycle goal (kg)
    predicted_goal = (base_weight +
                      training_age * 8 +
                      sets * 3 +
                      reps * 2 +
                      (intensity - 7) * 5 +
                      motor_control * 1.5)

    # Cycle length and periodization style from mappings
    cycle_length = np.array([cycle_length_mapping[cat] for cat in categories])
    periodization = np.array([periodization_mapping[cat] for cat in categories])

    # Weekly progression (kg/week) = (predicted_goal - base_weight) / cycle_length
    weekly_progression = (predicted_goal - base_weight) / cycle_length

    data = pd.DataFrame({
        'training_age' : training_age,
        'sets' : sets,
        'reps' : reps,
        'intensity' : intensity,
        'base_weight' : base_weight,
        'tempo' : tempo,
        'training_frequency' : training_frequency,
        'motor_control' : motor_control,
        'category' : categories,  # target: classification (0-4)
        'predicted_goal' : predicted_goal,  # target: regression (kg)
        'cycle_length' : cycle_length,  # target: regression (weeks)
        'periodization' : periodization,  # target: classification (0: linear, 1: DUP, 2: block)
        'weekly_progression' : weekly_progression  # target: regression (kg/week)
    })

    return data
