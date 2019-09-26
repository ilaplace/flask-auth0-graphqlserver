from quart import _request_ctx_stack
from models import User, Classifier, Patient, Feature, db
import pandas as pd
import numpy as np
import concurrent.futures
import joblib
import sys

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

    r = db.session.query(Patient, Feature).outerjoin(
        Feature, Patient.id == Feature.patient_id).all()

    df = np.arange(numberOfPatients*classifier.numberOfFeatureTypes, dtype=np.float).reshape(numberOfPatients, classifier.numberOfFeatureTypes)

    #a = np.arange(15).reshape(5, 3)
    for element in df.flat:
        df.flat[int(element)] = np.float(r[int(element)].Feature.featureValue)

    #np.set_printoptions(threshold=sys.maxsize)
    print(df)
    classifier = Classifier.query.filter_by(user_id=user_id).first()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        trainedClassifier = await loop.run_in_executor(pool, train, classifier,df,user_id)
        classifier.classifierStatus = trainedClassifier.classifierStatus
        db.session.add(classifier)
        db.session.commit()
        return "success"

# simulating a CPU bound task
def train(classifier, df, user_id):
    #print(df)
    sum(i * i for i in range(10 ** 7))
    vari = 123
    joblib.dump(vari, user_id+'.pkl')
    classifier.classifierStatus = "trained"
    print("Done training")
    return classifier

