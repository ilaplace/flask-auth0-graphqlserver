from functools import wraps
from os import environ as env
from dotenv import load_dotenv, find_dotenv
from quart import Quart, request, jsonify, _request_ctx_stack, flash, make_response
from quart_cors import cors, route_cors
import quart.flask_patch
from ariadne import QueryType, graphql, make_executable_schema, MutationType
from ariadne.constants import PLAYGROUND_HTML
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
import enum

import time
from graphServer import blueprint
from auth_helper import get_token_auth_header, requires_auth, AuthError
from models import db, Classifier, Feature, Patient, User


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = {'xlsx'}

APP = Quart(__name__)
APP.register_blueprint(blueprint)

# On heroku the environement variable IS_HEROKU becomes true
is_prod = os.environ.get('IS_HEROKU', None)
if is_prod:
    APP.config.from_object("config.ProductionConfig")
else:
    APP.config.from_object("config.DevelopmentConfig")

ORIGIN_URI = APP.config["FRONTEND_URI"]

print(f'Production: {is_prod}')

APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#db = SQLAlchemy(APP)
db.init_app(APP)
APP = cors(APP,
           allow_origin=ORIGIN_URI,
           allow_methods='*',
           allow_headers='*',
           allow_credentials=True
           )

class PatientStatus(enum.Enum):
    DIAGNOSED = 1
    FAILED = 2
    UNDIAGNOSE = 3


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Format error response and append status code.
@APP.errorhandler(AuthError)
def handle_auth_error(ex):
    response = jsonify(ex.error)
    response.status_code = ex.status_code
    return response


# Controllers API
@APP.route("/api/public")
def public():
    """No access token required to access this route
    """
    response = "Hello from a public endpoint! You don't need to be authenticated to see this."
    return jsonify(message=response)


@APP.route("/api/private")
@requires_auth
def private():
    """A valid access token is required to access this route
    """
    response = "Hello from a private endpoint! You need to be authenticated to see this."
    return jsonify(message=response)


# unused
@APP.route("/api/private-scoped")
@requires_auth
def private_scoped():
    """A valid access token and an appropriate scope are required to access this route
    """
    if requires_scope("read:messages"):
        response = "Hello from a private endpoint! You need to be authenticated and have a scope of read:messages to see this."
        return jsonify(message=response)
    raise AuthError({
        "code": "Unauthorized",
        "description": "You don't have access to this resource"
    }, 403)



# TODO: Use the domain environment variable
# TODO: Make sure the file name is unique
# TODO: Since this is an API don't flash but send response


@route_cors(
    allow_origin=ORIGIN_URI,
    allow_methods=["*"],
    allow_headers='*')
@APP.route('/api/diagnose', methods=['POST'])
@requires_auth
async def diagnose():
    data = await request.data
    print(data)
    return "success", 200


@route_cors(
    allow_origin=ORIGIN_URI,
    allow_methods=["*"],
    allow_headers='*')
@APP.route('/api/upload', methods=['POST'])
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
        file_path = os.path.join(APP.config['UPLOAD_FOLDER'], filename)
        user_id = _request_ctx_stack.top.current_user.get('sub')

        # this is not necessary as the file ereased
        # if os.path.exists(file_path):
        #     await importDatabase(filename, user_id)
        #     return jsonify(message="file already exists"), 200

        file.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))

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

# Create a SQLDatabase from the uploaded excel file
# This is just for a specific database
# TODO: Surround with try catch
# TODO: This needs to go to a seperate thread


def importDatabase(filename, user):
    df = pd.read_excel(os.path.join(APP.config['UPLOAD_FOLDER'], filename))

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

    os.remove(os.path.join(APP.config['UPLOAD_FOLDER'], filename))

