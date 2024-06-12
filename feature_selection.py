import numpy as np
import time
import csv

# Function to print the set of features
def format_feature_set(features):
    return "{" + ",".join(map(str, features)) + "}"

# Function which finds the class of the nearest neighbor of the given element using the given set of features
def get_nearest_neighbor_class(data, x, features):
    min_index = np.argmin(np.sum((data[:,features] - x[features])**2, axis=1))
    return data[min_index][0]

# Function to find the mean accuracy using "leaving-one-out" evaluation. 
def nearest_neighbor_accuracy(data, features):
    nearest_neighbor_classes = [get_nearest_neighbor_class(np.delete(data, i, axis=0), data[i], features) for i in range(len(data))]
    return np.mean(data[:,0] == nearest_neighbor_classes)

# Forward selection algorithm
def forward_selection(data, file_name):
    selected_features = [] # To store the set of features selected at the previous round
    previous_level_accuracy = None # To store the highest accuracy at the previous round
    highest_accuracy_features = None # To store the set of features which has the highest accuracy among all the rounds
    highest_accuracy = 0 # To store the highest accuracy among all the rounds
    print("Running nearest neighbor algorithm with individual features selected, using \"leaving-one-out\" evaluation\n")
    
    # Until we add all the features, do the following
    while len(selected_features) < data.shape[1] - 1:
        highest_accuracy_at_current_level = 0 # To store the highest accuracy in the current round
        highest_accuracy_features_at_current_level = None # To store the features having highest accuracy in the current round

        # Find the set of features which can be added to the selected features list
        remaining_features = list(set(range(1, data.shape[1])) - set(selected_features))

        # For each feature which can be added, find the accuracy when the feature is added to the selected set of features
        for feature in remaining_features:
            new_features = selected_features + [feature]
            new_accuracy = nearest_neighbor_accuracy(data, new_features)
            print(f"From the feature set {format_feature_set(new_features)}, accuracy is {round(new_accuracy*100, 2)}%")

            # If the current accuracy is higher than the highest in the round,
            # Record both the accuracy and the features selected as the highest in the round
            if new_accuracy > highest_accuracy_at_current_level:
                highest_accuracy_at_current_level = new_accuracy
                highest_accuracy_features_at_current_level = new_features
        
        # If the highest accuracy at the current level decreased when compared to the highest in the previous level,
        # Print a warning message, telling it could be a local maxima
        if previous_level_accuracy != None and highest_accuracy_at_current_level < previous_level_accuracy:
            print(f"(Warning! Accuracy has decreased! Continuing search in case of local maxima)")
        
        print(f"Selecting feature set {format_feature_set(highest_accuracy_features_at_current_level)} with accuracy {round(highest_accuracy_at_current_level*100, 2)}%\n")

        # Output the features selected and its accuracy at each round for further evaluation
        with open(file_name, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([format_feature_set(highest_accuracy_features_at_current_level), round(highest_accuracy_at_current_level*100, 2)])

        # If the highest accuracy at the round is higher than the highest accuracy among all the rounds,
        # Record both the accuracy and the features selected as the highest among all the rounds
        if highest_accuracy_at_current_level > highest_accuracy:
            highest_accuracy = highest_accuracy_at_current_level
            highest_accuracy_features = highest_accuracy_features_at_current_level
        
        selected_features = highest_accuracy_features_at_current_level
        previous_level_accuracy = highest_accuracy_at_current_level

    print("Finished Search!")
    print(f"The best feature set found was {format_feature_set(highest_accuracy_features)} with accuracy {round(highest_accuracy*100, 2)}%")


# Backward elimination algorithm
def backward_elimination(data, file_name):
    selected_features = list(range(1, data.shape[1])) # To store the set of features selected at the previous round
    previous_level_accuracy = None # To store the highest accuracy at the previous round
    highest_accuracy_features = list(range(1, data.shape[1])) # To store the set of features which has the highest accuracy among all the rounds
    highest_accuracy = nearest_neighbor_accuracy(data, selected_features) # To store the highest accuracy among all the rounds
    print("Running nearest neighbor algorithm with all features selected, using \"leaving-one-out\" evaluation")
    print(f"Initial accuracy with all the features selected is {round(highest_accuracy*100, 2)}%\n")

    # Until we have some features to be removed from the selected features, do the following
    while len(selected_features) > 1:
        highest_accuracy_at_current_level = 0 # To store the highest accuracy in the current round
        highest_accuracy_features_at_current_level = None # To store the features having highest accuracy in the current round
        
        # For each feature which can be removed from the selected features, find the accuracy when the feature is removed from the selected set of features
        for feature_index in range(len(selected_features)):
            new_features = np.delete(selected_features, feature_index)
            new_accuracy = nearest_neighbor_accuracy(data, new_features)
            print(f"From the feature set {format_feature_set(new_features)}, accuracy is {round(new_accuracy*100, 2)}%")
            
            # If the current accuracy is higher than the highest in the round,
            # Record both the accuracy and the features selected as the highest in the round
            if new_accuracy > highest_accuracy_at_current_level:
                highest_accuracy_at_current_level = new_accuracy
                highest_accuracy_features_at_current_level = new_features
        
        # If the highest accuracy at the current level decreased when compared to the highest in the previous level,
        # Print a warning message, telling it could be a local maxima
        if previous_level_accuracy != None and highest_accuracy_at_current_level < previous_level_accuracy:
            print(f"(Warning! Accuracy has decreased! Continuing search in case of local maxima)")
        
        print(f"Selecting feature set {format_feature_set(highest_accuracy_features_at_current_level)} with accuracy {round(highest_accuracy_at_current_level*100, 2)}%\n")

        # Output the features selected and its accuracy at each round for further evaluation
        with open(file_name, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([format_feature_set(highest_accuracy_features_at_current_level), round(highest_accuracy_at_current_level*100, 2)])
        
        # If the highest accuracy at the round is higher than the highest accuracy among all the rounds,
        # Record both the accuracy and the features selected as the highest among all the rounds
        if highest_accuracy_at_current_level > highest_accuracy:
            highest_accuracy = highest_accuracy_at_current_level
            highest_accuracy_features = highest_accuracy_features_at_current_level
        
        selected_features = highest_accuracy_features_at_current_level
        previous_level_accuracy = highest_accuracy_at_current_level
    
    print("Finished Search!")
    print(f"The best feature set found was {format_feature_set(highest_accuracy_features)} with accuracy {round(highest_accuracy*100, 2)}%")




def main():
    data_selection = {
        "1": "CS205_small_Data__4.txt",
        "2": "CS205_large_Data__14.txt"
    }

    print("Welcome to Feature Selection Algorithm!\n")

    print("Please select data. Enter one of the choice below")
    print("1: Small Dataset")
    print("2: Large Dataset")
    data_choice = input()
    if data_choice == "1" or data_choice == "2":
        data_file = data_selection[data_choice]
    else:
        print("Invalid Input!")
        return
    
    # Load the requested data from the txt file
    data = np.loadtxt(data_file, dtype=float)

    print("\nPlease select one of the following algorithm")
    print("1: Forward Selection")
    print("2: Backward Elimination")
    algorithm_choice = input()

    print(f"\nThe dataset has {data.shape[1]-1} features(excluding the class attribute), with {data.shape[0]} instances\n")

    start = time.time()
    if algorithm_choice == "1":
        forward_selection(data, data_choice+"forward.csv")
    elif algorithm_choice == "2":
        backward_elimination(data, data_choice+"backward.csv")
    else:
        print("Invalid Input!")
        return
    end = time.time()
    print(f"Running time for the algorithm: {round(end - start, 2)}sec")

main()
