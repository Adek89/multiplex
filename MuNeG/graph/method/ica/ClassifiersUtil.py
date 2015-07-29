__author__ = 'Adek'
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def giveNaiveBayes():
    return GaussianNB()

def giveKNearestNeighbours(nrOfNeighbours):
    return KNeighborsClassifier(nrOfNeighbours)

def giveLabelPropagation():
    return LabelPropagation()

def giveSVM():
    return SVC()

def giveDecisionTree():
    return DecisionTreeClassifier()

def knownModels(neighbors=2):
    return list({giveDecisionTree(), giveKNearestNeighbours(neighbors), giveLabelPropagation(), giveNaiveBayes(), giveSVM()})


