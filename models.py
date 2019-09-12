from flask_sqlalchemy import SQLAlchemy
import enum

db = SQLAlchemy()

class PatientStatus(enum.Enum):
    DIAGNOSED = 1
    FAILED = 2
    UNDIAGNOSE = 3


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
    diagnose = db.Column(db.String, nullable=True)
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

class Summation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(35))
    sum = db.Column(db.Integer)

    def __repr__(self):
        return '<Sum %d>' % self.sum