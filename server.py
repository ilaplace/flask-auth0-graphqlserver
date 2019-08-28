"""Forked from Python Flask API Auth0 integration example
"""

from functools import wraps
import json
from os import environ as env
from six.moves.urllib.request import urlopen

from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify, _request_ctx_stack, flash 
from flask_cors import cross_origin, CORS
from jose import jwt

from ariadne import QueryType, graphql_sync, make_executable_schema, MutationType, upload_scalar
from ariadne.constants import PLAYGROUND_HTML

from flask_sqlalchemy import SQLAlchemy

import os
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np

import enum
import flask_excel as excel


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
AUTH0_DOMAIN = env.get("AUTH0_DOMAIN")
API_IDENTIFIER = env.get("API_IDENTIFIER")
ALGORITHMS = ["RS256"]

UPLOAD_FOLDER = os.getcwd() + '/uploads'
ALLOWED_EXTENSIONS = {'xlsx'}
APP = Flask(__name__)
APP.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = SQLAlchemy(APP)
excel.init_excel(APP)

APP.secret_key = 'super secret key'
APP.config['SESSION_TYPE'] = 'filesystem'


#deneme (turns out it's necesseay)
CORS(APP, support_creadentials=True)
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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    status = db.Column(db.String, nullable=False)
    features = db.relationship('Feature', backref='patient',lazy=True)
    def __repr__(self):
        return self.status

class Feature(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    featureName = db.Column(db.String(20),nullable=False)
    featureValue=db.Column(db.String(20),nullable=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'))
    classifier_id = db.Column(db.Integer, db.ForeignKey('classifier.id'))

class Classifier(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    classifierStatus = db.Column(db.String, nullable=True)
    featrues = db.relationship('Feature', backref='classifier',lazy=True)
    

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
@cross_origin(headers=["Content-Type", "Authorization"])
def public():
    """No access token required to access this route
    """
    response = "Hello from a public endpoint! You don't need to be authenticated to see this."
    return jsonify(message=response)




@APP.route("/api/private")
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def private():
    """A valid access token is required to access this route
    """
    response = "Hello from a private endpoint! You need to be authenticated to see this."
    return jsonify(message=response)


#unused 
@APP.route("/api/private-scoped")
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
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
@cross_origin(headers=["Content-Type", "Authorization"])
def graphql_playgroud():
    # On GET request serve GraphQL Playground
    # You don't need to provide Playground if you don't want to
    # but keep on mind this will not prohibit clients from
    # exploring your API using desktop GraphQL Playground app.
    return PLAYGROUND_HTML, 200

# TODO: update route to /api/graphql
@APP.route("/graphql", methods=["POST"])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def graphql_server():
    # GraphQL queries are always sent as POST
    data = request.get_json()

    # Note: Passing the request to the context is optional.
    # In Flask, the current request is always accessible as flask.request
    success, result = graphql_sync(
        schema,
        data,
        context_value=request,
        debug=APP.debug
    )

    status_code = 200 if success else 400
    return jsonify(result), status_code

# TODO: Use the domain environment variable
# TODO: Make sure the file name is unique

@APP.route('/api/upload', methods=['POST'])
@cross_origin(headers=["Content-Type", "Authorization"])
@cross_origin(headers=["Access-Control-Allow-Origin", "http://localhost:3000"])
@requires_auth
def upload_file():
    status_code = 400 
    message = "unkown file"
    if 'file' not in request.files:
        flash('No file part')
    file = request.files['file']
    if file.filename == '':
        flash('No file selected!')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        message = filename
        file.save(os.path.join(APP.config['UPLOAD_FOLDER'], filename))
        
        user_id = _request_ctx_stack.top.current_user.get('sub')
        #Get the mail from the access token which is modified by the help of the help of the auth0 rules
        mail = _request_ctx_stack.top.current_user.get('https://dev-yy8du86w.eu/mail')
        

        this_user = User(id=user_id, mail=mail)
        df = pd.read_excel(os.path.join(APP.config['UPLOAD_FOLDER'],filename))
        
    
        db.session.add(this_user)
        db.session.commit()
        print(df['A'])
        status_code = 200 
    
    return jsonify(message=message), status_code


if __name__ == "__main__":
    type_defs = """
        type Query {
            hello: String!
        }
        type Mutation {
            sum(a: Int!, b: Int!): Int!
            uploadImage(file: Upload!): Boolean!
        }     
        type File {
            id: ID!
            path: String!
            filename: String!
            mimetype: String!
            encoding: String!
            }
        scalar Upload
        
    """

    query = QueryType()
    mutation = MutationType()

    @query.field("hello")
    def resolve_hello(_, info):
        request = info.context
        user_agent = request.headers.get("User-Agent","Guest")
        muser= _request_ctx_stack.top.current_user.get('sub')
        #return "Hello, %s" % request.headers
        return  Summation.query.filter_by(user=muser).first()

    @mutation.field("sum")
    def resolve_sum(_, info, a, b):
        c = a + b
        # to create a new record of summation 
        muser = _request_ctx_stack.top.current_user.get('sub')
        mus = Summation.query.filter_by(user=muser).first()
        sumasyon = Summation(user=muser, sum=c)
        
        # modify the existing record
        #mus.sum = c 
        #db.session.add(mus)
       
        db.session.add(sumasyon)
        db.session.commit()

        return c

    schema = make_executable_schema(type_defs, [query, mutation, upload_scalar])

    APP.run(port=env.get("PORT", 3010), debug=True)
