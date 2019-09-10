from quart import Blueprint, jsonify, _request_ctx_stack, current_app, request
from quart_cors import route_cors
from ariadne import QueryType, graphql, make_executable_schema, MutationType
import os
import asyncio
from auth_helper import requires_auth
from ariadne.constants import PLAYGROUND_HTML
from models import Classifier, Feature, Patient, User, db
from initializeClassifier import initializeClassifier

blueprint_graph = Blueprint('GraphQLServer', __name__)


@blueprint_graph.route('/extra/')
def extra():
    print("ekstra mk")
    response = "extra mkmk"
    return jsonify(message=response)


type_defs = """
    type Query {
        hello: String!
        checkStatus: String!
        getClassifier: Classifier
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

#port = int(os.environ.get('PORT', 8000))
#APP.run(host=APP.config['SERVER'], port=port)
schema = make_executable_schema(type_defs, [query, mutation])



@blueprint_graph.route("/graphql", methods=["GET"])
def graphql_playgroud():
    # On GET request serve GraphQL Playground
    # You don't need to provide Playground if you don't want to
    # but keep on mind this will not prohibit clients from
    # exploring your API using desktop GraphQL Playground app.
    return PLAYGROUND_HTML, 200

# TODO: update route to /api/graphql


@blueprint_graph.route("/graphql", methods=["POST"])
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