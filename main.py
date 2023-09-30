import json
import numpy as np

from model_src.model_implementation import LogisticRegression
from model_src.model_training import X_test, y_test
from sklearn.metrics import classification_report


if __name__ == "__main__":
    with open("model_src/model_parameters.json", 'r', encoding='utf-8') as f:
        model_parameters = json.loads(f.read())

model = LogisticRegression()
model.weights = np.array(model_parameters['weights'])
model.bias = np.array(model_parameters['bias'])
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
