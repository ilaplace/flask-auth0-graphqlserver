class Config(object):
    DEBUG = False
    TESTING = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///test.db"
    secret_key = 'super secret key'
    SESSION_TYPE = 'filesystem'

class ProductionConfig(Config):
    SERVER = "127.0.0.1"
    FRONTEND_URI = "https://diagnostician-frontend.herokuapp.com"


class DevelopmentConfig(Config):
    DEBUG = True
    SERVER = "0.0.0.0"
    FRONTEND_URI = "http://localhost:3000"

class TestingConfig(Config):
    TESTING = True