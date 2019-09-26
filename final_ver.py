import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import RUSBoostClassifier
from imblearn.metrics import geometric_mean_score

def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)
    print('')

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

dataset = pd.read_excel('df_created_by_python.xlsx')
print(dataset.to_string())

X = dataset.drop(columns=['target'])
print(X)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
base_estimator = AdaBoostClassifier(n_estimators=10)
rusboost = RUSBoostClassifier(n_estimators=10, base_estimator=base_estimator)
rusboost.fit(X_train, y_train)
y_pred_rusboost = rusboost.predict(X_test)
print('RUSBoost classifier performance:')
print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
      .format(balanced_accuracy_score(y_test, y_pred_rusboost),
              geometric_mean_score(y_test, y_pred_rusboost)))
cm_rusboost = confusion_matrix(y_test, y_pred_rusboost)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_rusboost, classes=np.unique(dataset.target),
                      ax=ax[1], title='RUSBoost classifier')

plt.show()
