from sklearn import linear_model
from sklearn import neural_network
from sklearn import pipeline
import math
from scipy import stats
from IPython import embed

# binary only currently
# returns probability of prediction=1
# TODO probably should rename this NN
class NN:
    def __init__(self):
        self.classifier = linear_model.LogisticRegression()
        rbm = neural_network.BernoulliRBM(random_state=0, verbose=True)

        rbm.learning_rate = 0.03
        rbm.n_iter = 50
        rbm.n_components = 100
        self.classifier.C = 6000.0

        self.rbm_logit = pipeline.Pipeline(steps=[('rbm', rbm), ('logistic', self.classifier)])

    def train(self, features, targets):
        self.rbm_logit.fit(features, targets)

    def validate(self, features, targets):
        self.errors = []
        self.targets = targets
        self.predictions = []
        for i, _features in enumerate(features):
            target = targets[i]
            prediction = self.predict_for_single_vector(_features)
            self.predictions.append(prediction)
            if prediction == target:
                self.errors.append(0)
            else:
                self.errors.append(1)

    def predict(self, features):
        self.predictions = []
        for i, _features in enumerate(features):
            prediction = self.predict_for_single_vector(_features)
            self.predictions.append(prediction)

    def mse(self):
        sq_error = [math.pow(e,2) for e in self.errors]
        return sum(sq_error) / len(sq_error)

    def r(self):
        # embed()
        results = stats.linregress(self.targets, self.predictions)
        return results[2]

    def predict_for_single_vector(self, features):
        pred_coeffs = self.rbm_logit.predict_proba(features)[0]
        id_max = [i for i, j in enumerate(pred_coeffs) if j == max(pred_coeffs)]
        prediction = self.classifier.classes_[id_max][0]

        return prediction
