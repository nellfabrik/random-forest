import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# Function to bin the dependent variable
def bin_count_bi_1(count):
    # Binning the "COUNT_BI_1" variable into two categories: 0 and 1
    return 0 if count == 0 else 1


def import_data(csv_file_path, numeric_predictors, drop_low_zero_percentage=True):
    data = pd.read_csv(csv_file_path, low_memory=False)
    # Binning the dependent variable
    data['bike_killed_injured'] = data['COUNT_BI_1'].apply(bin_count_bi_1)
    # Replace '<Null>' with NaN
    data.replace('<Null>', float('nan'), inplace=True)

    # Drop rows with null values
    data.dropna(inplace=True)

    # Drop rows with low zero percentage if drop_low_zero_percentage is True
    if drop_low_zero_percentage:
        zero_percentage = (data[numeric_predictors] == 0).mean(axis=1)
        data = data[zero_percentage > 0]

    return data


def check_class_distribution(labels):
    # Function to check the class distribution of the target variable
    class_counts = labels.value_counts()
    class_distribution = class_counts / class_counts.sum()
    print("Class Distribution:")
    print(class_distribution)


def main():
    csv_file_path = "/Users/nell/Desktop/DF.csv"

    # Define response and predictors
    response = "bike_killed_injured"
    numeric_predictors = [
        "sky", "signboard", "road", "fence", "sidewalk", "streetlight", "mountain", "pole", "ground", "traffic light",
        "bench", "bridge", "trash can", "hill", "building_all", "tree_all", "vehicle", "greenspace", "grass",
        "plant, flora, plant life", "volumes", "RVMTPerP10", "2023 Population Density", "2023 Total Population",
        "2023 Median Household Income",
        "2021 HHs: Inc Below Poverty Level (ACS 5-Yr): Percent",
    ]

    data = import_data(csv_file_path, numeric_predictors, drop_low_zero_percentage=True)

    categorical_predictors = ["intersection", "weather", "ped_action", "road_surface", "lighting", "maxspeed", "iway",
                              "lanes", "ifc"]

    # Perform one-hot encoding for categorical variables
    data = pd.get_dummies(data, columns=categorical_predictors)

    # Splitting data into features and labels
    features = data[numeric_predictors + list(data.columns[data.columns.str.startswith('intersection_')])]
    labels = data[response]

    # Check class distribution after binning
    check_class_distribution(labels)

    # Split the data into training and testing sets with stratified sampling
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, stratify=labels, test_size=0.2, random_state=42
    )

    # Train and evaluate the model
    rf_classifier = RandomForestClassifier(
        min_samples_leaf=50,
        n_estimators=150,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=50,
        max_features='sqrt'
    )
    rf_classifier.fit(train_features, train_labels)

    # Evaluate the model on the training set
    train_predictions = rf_classifier.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_predictions)

    # Evaluate the model on the test set
    test_predictions = rf_classifier.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    # Feature importance
    importances = rf_classifier.feature_importances_
    feature_names = train_features.columns

    # Create a list of (feature, importance) pairs
    feature_importance_pairs = list(zip(feature_names, importances))

    # Sort the pairs based on importance (in descending order)
    sorted_feature_importance_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

    # Separate the sorted features and importances into separate lists
    sorted_features, sorted_importances = zip(*sorted_feature_importance_pairs)

    print("Feature Importances (Highest to Lowest):")
    for feature, importance in sorted_feature_importance_pairs:
        print(f"{feature}: {importance}")

    plt.figure(figsize=(10, 14))
    plt.barh(sorted_features, sorted_importances)  # horizontal barplot
    plt.title('Random Forest Feature Importance (Highest to Lowest)',
              fontdict={'fontname': 'Comic Sans MS', 'fontsize': 20})
    plt.xlabel('Importance', fontdict={'fontsize': 12})
    plt.yticks(fontsize=10)
    plt.show()


if __name__ == "__main__":
    main()


