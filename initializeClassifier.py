from quart import _request_ctx_stack
from models import User, Classifier, Patient, Feature, db
import pandas as pd
import numpy as np
import concurrent.futures

async def initializeClassifier(loop):
    ''' Called from the startTraining mutation, caller of the train method'''
    user_id = _request_ctx_stack.top.current_user.get('sub')
    if not (User.query.get(user_id)):
        print("No database found")

    r = db.session.query(Patient, Feature).outerjoin(
        Feature, Patient.id == Feature.patient_id).all()

    a = np.arange(15).reshape(5, 3)
    for element in a.flat:
        a.flat[element] = np.int64(r[element].Feature.featureValue)
    print(a)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        trainedClassifier = await loop.run_in_executor(pool, train, classifier)
        classifier.classifierStatus = trainedClassifier.classifierStatus
        db.session.add(classifier)
        db.session.commit()
        return "success"

# simulating a CPU bound task
def train(classifier):
    sum(i * i for i in range(10 ** 6))

    classifier.classifierStatus = "done"
    print("doneeee")
    return classifier

