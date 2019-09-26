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



import time
from graphServer import blueprint_graph
from fileUpload import blueprint_upload
from auth_helper import get_token_auth_header, requires_auth, AuthError
from models import Classifier, Feature, Patient, User, db

import random

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

UPLOAD_FOLDER = os.getcwd()


APP = Quart(__name__)
APP.register_blueprint(blueprint_graph)
APP.register_blueprint(blueprint_upload)


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

# Initialization of the database
db.init_app(APP)
db.app = APP
db.create_all()

APP = cors(APP, allow_origin=ORIGIN_URI,
                allow_methods='*',
                allow_headers='*',
                allow_credentials=True)


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


@route_cors(
    allow_origin=ORIGIN_URI,
    allow_methods=["*"],
    allow_headers='*')
@APP.route('/api/diagnose', methods=['POST'])
@requires_auth
async def diagnose():
    data = await request.data
    print(data)
    datastr = str(data)

    # Read the number of patients
    index = datastr.find('numberOfPatients')
    if(index == -1):
        return jsonify(message="numberOfPatients could not be read")
    numberOfPatients = int(datastr[index+len('numberOfPatients')+2])
    resp = []
    for x in range(numberOfPatients):
        resp.append(random.randint(1,10))

    return jsonify(message=resp)


