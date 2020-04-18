import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from csv file, split into test and train sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    # Reading data in from csv file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        for row in reader:
            data.append({
                "evidence": [[int(cell) for cell in row[0]], [str(cell) for cell in row[1]],
                             [int(cell) for cell in row[2]], [str(cell) for cell in row[3]],
                             [int(cell) for cell in row[4]], [float(cell) for cell in row[5::9]],
                             [str(cell) for cell in row[9]], [str(cell) for cell in row[10]],
                             [int(cell) for cell in row[11::16]]],
                "label": "Purchase" if row[16] == "1" else "No Purchase"
            })

        return data[0], data[1]


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    X_training = [row["evidence"] for row in evidence]
    y_training = [row["label"] for row in labels]

    model.fit(X_training, y_training)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).
    """
    correct = 0
    incorrect = 0
    total = 0

    for actual, predicted in zip(labels, predictions):
        total += 1
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

    sensitivity = float(correct / total)
    specificity = float(incorrect / total)
    return sensitivity, specificity


if __name__ == "__main__":
    main()
