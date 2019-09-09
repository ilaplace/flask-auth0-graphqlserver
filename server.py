"""Forked from Python Flask API Auth0 integration example
"""

from functools import wraps
import json
from os import environ as env
from six.moves.urllib.request import urlopen
from dotenv import load_dotenv, find_dotenv
from quart import Quart, request, jsonify, _request_ctx_stack, flash, make_response
from quart_cors import cors, route_cors
import quart.flask_patch
from jose import jwt
from ariadne import QueryType, graphql, make_executable_schema, MutationType, upload_scalar, SubscriptionType
from ariadne.constants import PLAYGROUND_HTML
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import enum
import asyncio

import concurrent.futures
from serversentevent import ServerSentEvent

import time

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
AUTH0_DOMAIN = env.get("AUTH0_DOMAIN")
API_IDENTIFIER = env.get("API_IDENTIFIER")
ALGORITHMS = ["RS256"]

UPLOAD_FOLDER = os.getcwd() + '/uploads'

ALLOWED_EXTENSIONS = {'xlsx'}
APP = Quart(__name__)

is_prod = os.environ.get('IS_HEROKU', None)
if is_prod:
    APP.config.from_object("config.ProductionConfig")
else:
    APP.config.from_object("config.DevelopmentConfig")

ORIGIN_URI = APP.config["FRONTEND_URI"]

print(f'ENV is set to: {APP.config["ENV"]}')

APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(APP)



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


class User(db.Model):
    id = db.Column(db.String(20), primary_key=True)
    mail = db.Column(db.String(50), nullable=False)
    patients = db.relationship('Patient', backref='user', lazy=True)
    classifiers = db.relationship('Classifier', backref='user', lazy=True)

    def __repr__(self):
        return self.mail


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(20), db.ForeignKey('user.id'))
    status = db.Column(db.String, nullable=False)
    features = db.relationship('Feature', backref='patient', lazy=False)

    def __repr__(self):
        return self.id


class Feature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    featureName = db.Column(db.String(20), nullable=False)
    featureValue = db.Column(db.String(20), nullable=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    classifier_id = db.Column(db.Integer, db.ForeignKey('classifier.id'))

    def __repr__(self):
        return self.featureValue


class Classifier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    classifierStatus = db.Column(db.String, nullable=True)
    features = db.relationship(
        'Feature', backref='classifier', lazy=True, cascade="delete")
    numberOfFeatureTypes = db.Column(db.Integer)

    def __repr__(self):
        return self.id


# Example table
class Summation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(35))
    sum = db.Column(db.Integer)

    def __repr__(self):
        return '<Sum %d>' % self.sum

# Format error response and append status code.


class AuthError(Exception):
    def __init__(self, error, status_code):
        self.error = error
        self.status_code = status_code


@APP.errorhandler(AuthError)
def handle_auth_error(ex):
    response = jsonify(ex.error)
    response.status_code = ex.status_code
    return response


def get_token_auth_header():
    """Obtains the access token from the Authorization Header
    """
    auth = request.headers.get("Authorization", None)
    if not auth:
        raise AuthError({"code": "authorization_header_missing",
                         "description":
                         "Authorization header is expected"}, 401)

    parts = auth.split()

    if parts[0].lower() != "bearer":
        raise AuthError({"code": "invalid_header",
                         "description":
                         "Authorization header must start with"
                         " Bearer"}, 401)
    elif len(parts) == 1:
        raise AuthError({"code": "invalid_header",
                         "description": "Token not found"}, 401)
    elif len(parts) > 2:
        raise AuthError({"code": "invalid_header",
                         "description":
                         "Authorization header must be"
                         " Bearer token"}, 401)

    token = parts[1]
    return token


def requires_scope(required_scope):
    """Determines if the required scope is present in the access token
    Args:
        required_scope (str): The scope required to access the resource
    """
    token = get_token_auth_header()
    unverified_claims = jwt.get_unverified_claims(token)
    if unverified_claims.get("scope"):
        token_scopes = unverified_claims["scope"].split()
        for token_scope in token_scopes:
            if token_scope == required_scope:
                return True
    return False


def requires_auth(f):
    """Determines if the access token is valid
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_auth_header()
        jsonurl = urlopen("https://"+AUTH0_DOMAIN+"/.well-known/jwks.json")
        jwks = json.loads(jsonurl.read())
        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.JWTError:
            raise AuthError({"code": "invalid_header",
                             "description":
                             "Invalid header. "
                             "Use an RS256 signed JWT Access Token"}, 401)
        if unverified_header["alg"] == "HS256":
            raise AuthError({"code": "invalid_header",
                             "description":
                             "Invalid header. "
                             "Use an RS256 signed JWT Access Token"}, 401)
        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
        if rsa_key:
            try:
                payload = jwt.decode(
                    token,
                    rsa_key,
                    algorithms=ALGORITHMS,
                    audience=API_IDENTIFIER,
                    issuer="https://"+AUTH0_DOMAIN+"/"
                )
            except jwt.ExpiredSignatureError:
                raise AuthError({"code": "token_expired",
                                 "description": "token is expired"}, 401)
            except jwt.JWTClaimsError:
                raise AuthError({"code": "invalid_claims",
                                 "description":
                                 "incorrect claims,"
                                 " please check the audience and issuer"}, 401)
            except Exception:
                raise AuthError({"code": "invalid_header",
                                 "description":
                                 "Unable to parse authentication"
                                 " token."}, 401)

            _request_ctx_stack.top.current_user = payload
            return f(*args, **kwargs)
        raise AuthError({"code": "invalid_header",
                         "description": "Unable to find appropriate key"}, 401)
    return decorated


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


@APP.route("/graphql", methods=["GET"])
def graphql_playgroud():
    # On GET request serve GraphQL Playground
    # You don't need to provide Playground if you don't want to
    # but keep on mind this will not prohibit clients from
    # exploring your API using desktop GraphQL Playground app.
    return PLAYGROUND_HTML, 200

# TODO: update route to /api/graphql


@route_cors(
    allow_origin=ORIGIN_URI,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"])
@APP.route("/graphql", methods=["POST"])
@requires_auth
async def graphql_server():
    # GraphQL queries are always sent as POST
    data = await request.get_json()

    # Note: Passing the request to the context is optional.
    # In Flask, the current request is always accessible as flask.request

    success, result = await graphql(
        schema,
        data,
        context_value=request
    )

    status_code = 200 if success else 400
    return jsonify(result), status_code

# TODO: Use the domain environment variable
# TODO: Make sure the file name is unique
# TODO: Since this is an API don't flash but send response


@route_cors(
    allow_origin=ORIGIN_URI,
    allow_methods=["POST", "GET"],
    allow_headers='*')
@APP.route('/api/diagnose', methods=['POST'])
@requires_auth
async def diagnose():
    data = await request.data
    print(data)
    return "success", 200


@route_cors(
    allow_origin=ORIGIN_URI,
    allow_methods=["POST", "GET"],
    allow_headers='*')
@APP.route('/api/upload', methods=['POST'])
@requires_auth
async def upload_file():
    message = "unkown file"
    if 'file' not in await request.files:
        flash('No file part')
    file = (await request.files)['file']
    if file.filename == '':
        flash('No file selected!')

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


# simulating a CPU bound task
def train(classifier):
    sum(i * i for i in range(10 ** 6))

    classifier.classifierStatus = "done"
    print("doneeee")
    return classifier


# Initialize a classifier from the all available features
# TODO: User must be blocked from asking to training multiple times
async def initializeClassifier(loop):
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


if __name__ == "__main__":
    type_defs = """
        type Query {
            hello: String!
            checkStatus: String!
            getClassifier: Classifier!
            }
    
        type Mutation {
            sum(a: Int!, b: Int!): Int!
            startTraining: String!
            deleteDatabase: String!
            } 

        type Feature {
            id: ID!
            featureName: String!
            featureValue: String!
            patient_id: String!
            classifier_id: String!
            } 

        type Classifier {
            id: ID!
            user_id: ID!
            classifierStatus: String
            features: [Feature]
            featureTypes: [String]
            }
    """

    query = QueryType()
    mutation = MutationType()

    @mutation.field("deleteDatabase")
    def resolve_delete_database(_, info):
        user_id = _request_ctx_stack.top.current_user.get('sub')
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        if classifier is None:
            return "failed"
        r = Classifier.query.get(classifier.id)
        p = Patient.query.filter_by(user_id=user_id).all()
        for patient in p:
            db.session.delete(patient)
        
        #db.session.delete(p)
        db.session.delete(r)
        db.session.commit()
        return "deleted"

    # Return how much feature type user has
    @query.field("getClassifier")
    def resolve_get_classifier(_, info):
        user_id = _request_ctx_stack.top.current_user.get('sub')
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        if classifier is None:
            print("failed to get the classifer")
            return "failed"
        # Add payload the distinct features array so that the table can be constructed accordingly
        r = Feature.query.with_entities(Feature.featureName).filter_by(
            classifier_id=classifier.id).distinct()
        classifier.featureTypes = r
        print(classifier.classifierStatus)
        return classifier

    @query.field("hello")
    def resolve_hello(_, info):
        request = info.context
        #user_agent = request.headers.get("User-Agent","Guest")
        muser = _request_ctx_stack.top.current_user.get('sub')
        # return "Hello, %s" % request.headers
        return "yello"

    @query.field("checkStatus")
    def resolve_chech_status(_, info):
        user_id = _request_ctx_stack.top.current_user.get('sub')
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        if classifier is None:
            return "failed"
        print(classifier.classifierStatus)
        return classifier.classifierStatus

    @mutation.field("sum")
    def resolve_sum(_, info, a, b):
        c = a + b
        # to create a new record of summation
        muser = _request_ctx_stack.top.current_user.get('sub')
        mus = Summation.query.filter_by(user=muser).first()
        sumasyon = Summation(user=muser, sum=c)

        # modify the existing record
        #mus.sum = c
        # db.session.add(mus)

        db.session.add(sumasyon)
        db.session.commit()

        return c

    @mutation.field("startTraining")
    async def resolve_train(_, info):
        user_id = _request_ctx_stack.top.current_user.get('sub')
        classifier = Classifier.query.filter_by(user_id=user_id).first()
        # If the classifier is not in training train it
        if classifier.classifierStatus == "untrained":
            classifier.classifierStatus = "training"
        #db.session.add(classifier)
        #db.session.commit()
        loop = asyncio.get_running_loop()
        result = await initializeClassifier(loop)
        return classifier.classifierStatus

    
    schema = make_executable_schema(type_defs, [query, mutation])
    APP.run(host=APP.config['SERVER'], port=env.get("PORT", 8000))
