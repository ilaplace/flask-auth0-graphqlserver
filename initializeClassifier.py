from quart import _request_ctx_stack
from models import User, Classifier, Patient, Feature, db
import pandas as pd
import numpy as np
import concurrent.futures
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import RUSBoostClassifier
from imblearn.metrics import geometric_mean_score

async def classifier_task():
    ''' Called from the startTraining mutation, caller of the train method'''
    user_id = _request_ctx_stack.top.current_user.get('sub')
    if not (User.query.get(user_id)):
        print("No database found")

    r = db.session.query(Patient, Feature).outerjoin(
        Feature, Patient.id == Feature.patient_id).all()

    a = np.arange(15).reshape(5, 3)
    for element in a.flat:
        a.flat[element] = np.int64(r[element].Feature.featureValue)


    classifier = Classifier.query.filter_by(user_id=user_id).first()
    trainedClassifier = train(classifier)
    classifier.classifierStatus = trainedClassifier.classifierStatus
    db.session.add(classifier)
    db.session.commit()
    return "success"

async def initializeClassifier(loop):
    ''' Called from the startTraining mutation, caller of the train method'''
    user_id = _request_ctx_stack.top.current_user.get('sub')

    classifier = Classifier.query.filter_by(user_id=user_id).first()
    patietns = Patient.query.filter_by(user_id=user_id)
    numberOfPatients = patietns.count()
    if not (User.query.get(user_id)):
        print("No database found")

    pat = Patient.query.filter_by(user_id=user_id).all()
    
    # y is the label matrix
    # TODO: Check if labels can be string or not
    y = np.arange(numberOfPatients, dtype=np.float)
    for i, index in enumerate(y):
        y[i] = pat[i].diagnose

    r = db.session.query(Patient, Feature).outerjoin(
        Feature, Patient.id == Feature.patient_id).all()

    # df is the data matrix X
    df = np.arange(numberOfPatients*classifier.numberOfFeatureTypes, dtype=np.float).reshape(numberOfPatients, classifier.numberOfFeatureTypes)

    for element in df.flat:
        df.flat[int(element)] = np.float(r[int(element)].Feature.featureValue)

    #np.set_printoptions(threshold=sys.maxsize)
    classifier = Classifier.query.filter_by(user_id=user_id).first()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        trainedClassifier = await loop.run_in_executor(pool, train, classifier,df,y,user_id)
        classifier.classifierStatus = trainedClassifier.classifierStatus
        db.session.add(classifier)
        db.session.commit()
        return "success"


def train(classifier, df,y, user_id):
    ''' The main training function that runs on a seperate process'''
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=0)
    base_estimator = AdaBoostClassifier(n_estimators=10)
    rusboost = RUSBoostClassifier(n_estimators=10, base_estimator=base_estimator)
    rusboost.fit(X_train, y_train)
    y_pred_rusboost = rusboost.predict(X_test)
    print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'.format(balanced_accuracy_score(y_test, y_pred_rusboost), geometric_mean_score(y_test, y_pred_rusboost)))
    cm_rusboost = confusion_matrix(y_test, y_pred_rusboost)
    joblib.dump(rusboost, user_id+'.pkl')
    classifier.classifierStatus = "trained"
    print("Done training")
    return classifier

