from data.data_preprocessing import X_scaled, y
from sklearn.model_selection import train_test_split
from model_src.model_implementation import LogisticRegression
import json

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=70)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = LogisticRegression(lr=0.01, n_iters=350, threshold=0.5)
model.fit(X_train, y_train)

model_parameters = {
    "weights": model.weights.tolist(),
    "bias": model.bias.tolist()
}
# print(model_parameters)

with open("model_parameters.json", 'w') as f:
    f.write(json.dumps(model_parameters))
