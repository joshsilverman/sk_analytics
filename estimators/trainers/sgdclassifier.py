from sklearn import linear_model
import math
from scipy import stats
from IPython import embed

# binary only currently
# returns probability of prediction=1
# TODO probably should rename this SGDBinaryClassifier
class SGDClassifier:
    def __init__(self):
        self.sdgclassifier_model = linear_model.SGDClassifier(loss="log", penalty="elasticnet",n_iter = 1000, alpha=.1)


    def train(self, features, targets):
        self.sdgclassifier_model.fit(features, targets)

    def validate(self, features, targets):
        self.errors = []
        self.targets = targets
        self.predictions = []
        for i, _features in enumerate(features):
            target = targets[i]
            prediction = self.predict_prob_of_prediction_1_for_single_vector(_features)
            self.predictions.append(prediction)
            self.errors.append(prediction - target)

    def predict(self, features):
        self.predictions = []
        for i, _features in enumerate(features):
            prediction = self.predict_prob_of_prediction_1_for_single_vector(_features)
            self.predictions.append(prediction)

    def mse(self):
        sq_error = [math.pow(e,2) for e in self.errors]
        return sum(sq_error) / len(sq_error)

    def r(self):
        # embed()
        results = stats.linregress(self.targets, self.predictions)
        return results[2]

    def predict_prob_of_prediction_1_for_single_vector(self, features):
        index_of_prediction_1 = list(self.sdgclassifier_model.classes_).index(1)
        prediction = self.sdgclassifier_model.predict_proba(features)[0][index_of_prediction_1]
        return prediction
