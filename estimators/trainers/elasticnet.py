from sklearn import linear_model
import math
from scipy import stats
from IPython import embed

class Elasticnet:
    def __init__(self):
        self.elasticnet_model = linear_model.ElasticNetCV()

    def train(self, features, targets):
        self.elasticnet_model.fit(features, targets)

    def validate(self, features, targets):
        self.errors = []
        self.targets = targets
        self.predictions = []
        for i, _features in enumerate(features):
            target = targets[i] or 1.0
            prediction = self.elasticnet_model.predict(_features)
            self.predictions.append(prediction)

            # normalize
            if prediction < 0:
                prediction = 0
            
            self.errors.append(prediction - target)

    def predict(self, features):
        self.predictions = []
        for i, _features in enumerate(features):
            prediction = self.elasticnet_model.predict(_features)

            # normalize
            if prediction < 0:
                prediction = 0

            self.predictions.append(prediction)

    def mse(self):
        sq_error = [math.pow(e,2) for e in self.errors]
        return sum(sq_error) / len(sq_error)

    def r(self):
        # embed()
        results = stats.linregress(self.targets, self.predictions)
        return results[2]
