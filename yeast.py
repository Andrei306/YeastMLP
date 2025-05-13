import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Data reading and processing
file_path = "yeast.data"  # Path

columns = ["Sequence_Name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "Class"]
data = pd.read_csv(file_path, sep=r"\s+", names=columns)

# Separating characteristics from tags
X = data.iloc[:, 1:-1].values # Characteristics
y = data.iloc[:, -1].values # Tag (Class)

# Converting tags in numerical format (MIT, NUC... -> 0, 1...)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Sharing datas (75% train - 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)

# Datas normalization (Datas are already normalized - just in case)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameters tuning
hidden_layer_options = [(8,), (16,), (32,), (8, 8), (16, 32), (32, 16)]
learning_rates = [0.1, 0.01]
activations = ["relu"] # Information processing - rectified linear unit
solvers = ["adam"] # Learning algorithm

best_accuracy = 0
best_params = None

# Testing all combination to find the best network

for layers in hidden_layer_options:
    for lr in learning_rates:
        for activation in activations:
            for solver in solvers:
                model = MLPClassifier(
                    hidden_layer_sizes=layers,
                    activation=activation,
                    solver=solver,
                    learning_rate_init=lr,
                    alpha=0.0001,  # L2 Regularization -> Prevents overfitting
                    max_iter=2000, # Go through train set 2000 times
                    batch_size=32, # Examples processed simultaneously at each learning step
                    early_stopping=True, # Stop, if not learning
                    random_state=50 # Same results each time
                )

                model.fit(X_train, y_train) #Train
                y_pred = model.predict(X_test) #Test
                accuracy = accuracy_score(y_test, y_pred) #Result

                results = []
                accuracies = []
                results.append((str(layers), lr, activation, solver, accuracy))
                accuracies.append(accuracy)

                print(f"Layers: {layers}, LR: {lr}, Activation: {activation}, Solver: {solver}, Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (layers, lr, activation, solver, accuracy, model)

# Best
print("\n Best combination:")
print(f"Layers: {best_params[0]}, Learning rate {best_params[1]}, Activation: {best_params[2]}, Solver: {best_params[3]}, Accuracy: {best_accuracy:.4f}")

# Confusion matrix for best model
best_model = best_params[5]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix - Best MLP Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout() 
plt.show()
