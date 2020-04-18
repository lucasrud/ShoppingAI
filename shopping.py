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
    """ TODO: load data from CSV and convert into a list of evidence lists and list of labels.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [int(cell) for cell in (row[0], float(cell)) for cell in (row[1], int(cell))
                         for cell in (row[2], float(cell)) for cell in (row[3], int(cell)) for cell in (row[4],
                         float(cell)) for cell in (row[5::9], int(cell)) for cell in (row[9::16])
                         ],
            "label": "Purchase" if row[16] == "1" else "No Purchase"
        })
    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
