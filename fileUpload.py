import os 
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from auth_helper import requires_auth
from quart import _request_ctx_stack, request, Blueprint, current_app, jsonify
from models import User, Feature, Classifier, Patient, db

ALLOWED_EXTENSIONS = {'xlsx'}

blueprint_upload = Blueprint('fileUpload', __name__)

@blueprint_upload.route('/api/upload', methods=['POST'])
@requires_auth
async def upload_file():
    message = "unkown file"
    if 'file' not in await request.files:
        return jsonify(message="No file part"), 400
    file = (await request.files)['file']
    if file.filename == '':
         return jsonify(message="No file Selected"), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        message = filename
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        user_id = _request_ctx_stack.top.current_user.get('sub')

        # this is not necessary as the file ereased
        # if os.path.exists(file_path):
        #     await importDatabase(filename, user_id)
        #     return jsonify(message="file already exists"), 200

        file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

        # TODO: Check if the user exists
        user_id = _request_ctx_stack.top.current_user.get('sub')
        # Get the mail from the access token which is modified by the help of the help of the auth0 rules
        mail = _request_ctx_stack.top.current_user.get(
            'https://dev-yy8du86w.eu/mail')

        this_user = User(id=user_id, mail=mail)
        # Do not create a user object if it already exists
        if not (User.query.get(user_id)):
            db.session.add(this_user)
            db.session.commit()

        classifier = Classifier(user_id=user_id, classifierStatus="untrained")

        # TODO: Do the counting before commiting
        db.session.add(classifier)
        db.session.commit()

        importDatabase(filename, user_id)

    return jsonify(message=message), 200


def importDatabase(filename, user):
    df = pd.read_excel(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

    for index, row in df.iterrows():
        new_patient = Patient(user_id=user, status="undiag")
        featureA = Feature(
            featureName='A', featureValue=str(row[0]), classifier_id=1)
        featureB = Feature(
            featureName='B', featureValue=str(row[1]), classifier_id=1)
        featureC = Feature(
            featureName='C', featureValue=str(row[2]), classifier_id=1)
        new_patient.features.append(featureA)
        new_patient.features.append(featureB)
        new_patient.features.append(featureC)
        db.session.add(new_patient)
        db.session.commit()

    os.remove(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS