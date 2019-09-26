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
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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


workbook_deneme = pd.ExcelFile('dataset_editted.xlsx')
sheet_names = workbook_deneme.sheet_names

# reading spread sheets
sheet_1 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[0])
sheet_1['target'] = sheet_1.index[0]
sheet_1['target_1'] = sheet_1.index[0]

sheet_2 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[1])
sheet_2['target'] = sheet_2.index[1]
sheet_2['target_1'] = sheet_1.index[1]

sheet_3 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[2])
sheet_3['target'] = sheet_3.index[2]
sheet_3['target_1'] = sheet_3.index[1]

sheet_4 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[3])
sheet_4['target'] = sheet_4.index[3]
sheet_4['target_1'] = sheet_3.index[1]

sheet_5 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[4])
sheet_5['target'] = sheet_5.index[4]
sheet_5['target_1'] = sheet_3.index[1]

sheet_6 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[5])
sheet_6['target'] = sheet_6.index[5]
sheet_6['target_1'] = sheet_3.index[1]

sheet_7 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[6])
sheet_7['target'] = sheet_7.index[6]
sheet_7['target_1'] = sheet_3.index[1]

sheet_8 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[7])
sheet_8['target'] = sheet_8.index[7]
sheet_8['target_1'] = sheet_3.index[1]

# sheet_9 = pd.read_excel('dataset_editted.xlsx', sheet_name= sheet_names[8])
# sheet_9['target'] = sheet_9.index[8]
# sheet_9['target_1'] = sheet_3.index[1]

# creating complete data frame
# dataset = pd.concat([sheet_1, sheet_2, sheet_3, sheet_4, sheet_5, sheet_6, sheet_7, sheet_8, sheet_9], ignore_index=True)
# dataset_seek = pd.concat([sheet_2, sheet_3, sheet_4, sheet_5, sheet_6, sheet_7, sheet_8, sheet_9], ignore_index=True)

dataset = pd.concat([sheet_1, sheet_2, sheet_3, sheet_4, sheet_5, sheet_6, sheet_7, sheet_8], ignore_index=True)
dataset_seek = pd.concat([sheet_2, sheet_3, sheet_4, sheet_5, sheet_6, sheet_7, sheet_8], ignore_index=True)


# dataset.columns = [''] * len(dataset.columns)
dataset = dataset.replace({'< ': ''}, regex=True)
dataset = dataset.replace({'<': ''}, regex=True)
dataset = dataset.replace({'\*': ' '}, regex=True)
dataset = dataset.replace({' ': np.nan}, regex=True)
dataset = dataset.replace({'-----': np.nan}, regex=True)
dataset = dataset.replace({',':'.'}, regex=True)
dataset = dataset.replace({'.':np.nan}, regex=True)

dataset_seek = dataset_seek.replace({'< ': ''}, regex=True)
dataset_seek = dataset_seek.replace({'<': ''}, regex=True)
dataset_seek = dataset_seek.replace({'\*': ' '}, regex=True)
dataset_seek = dataset_seek.replace({' ': np.nan}, regex=True)
dataset_seek = dataset_seek.replace({'-----': np.nan}, regex=True)
dataset_seek = dataset_seek.replace({',': '.'}, regex=True)
dataset_seek = dataset_seek.replace({'.': np.nan}, regex=True)

# replace missing values
dataset = dataset.fillna(0)
dataset_seek = dataset_seek.fillna(0)
print(dataset.to_string())
print(dataset_seek.to_string())

# print(dataset.isnull().sum())
# print(sheet_1.mean(), sheet_2.mean(),sheet_3.mean(),sheet_4.mean(),sheet_5.mean(),sheet_6.mean(),sheet_7.mean(),sheet_8.mean(),sheet_9.mean())
# print(sheet_1.mean() + sheet_2.mean() + sheet_3.mean()+sheet_4.mean()+sheet_5.mean()+sheet_6.mean()+sheet_7.mean()+sheet_8.mean()+sheet_9.mean())
dataset = dataset.astype(float)
dataset_seek = dataset_seek.astype(float)
# dataset['target'] = dataset.index[0]
# print(dataset['Aldosterone (ng/ml)']) # printing specific column


X = dataset.drop(columns=['target', 'Estrone (ng/ml)', 'Estradiol (ng/ml)', 'DHT (ng/ml)', 'Cortisone (ng/ml)', '21-Deoxycortisol  (ng/ml)'])
X_seek = dataset_seek.drop(columns=['target', 'Estrone (ng/ml)', 'Estradiol (ng/ml)', 'DHT (ng/ml)', 'Cortisone (ng/ml)', '21-Deoxycortisol  (ng/ml)'])
print(X)
y = dataset['target_1']
y_seek = dataset_seek['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
X_train_seek, X_test_seek, y_train_seek, y_test_seek = train_test_split(X_seek, y_seek, test_size=0.33, random_state=0)
base_estimator = AdaBoostClassifier(n_estimators=10)


# eec = EasyEnsembleClassifier(n_estimators=10,
#                              base_estimator=base_estimator,
#                              n_jobs=-1)
# eec.fit(X_train_seek, y_train_seek)
# y_pred_eec = eec.predict(X_test_seek)
# print('Easy ensemble classifier performance:')
# print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
#       .format(balanced_accuracy_score(y_test_seek, y_pred_eec),
#               geometric_mean_score(y_test_seek, y_pred_eec)))
# cm_eec = confusion_matrix(y_test_seek, y_pred_eec)
# fig, ax = plt.subplots(ncols=2)
# plot_confusion_matrix(cm_eec, classes=np.unique(dataset.target), ax=ax[0],
#                       title='Easy ensemble classifier')


base_estimator = AdaBoostClassifier(n_estimators=10)
rusboost = RUSBoostClassifier(n_estimators=10,
                              base_estimator=base_estimator)
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

rusboost.fit(X_train_seek, y_train_seek)

y_pred_rusboost_seek = rusboost.predict(X_test_seek)
print('RUSBoost classifier performance:')
print('Balanced accuracy: {:.2f} - Geometric mean {:.2f}'
      .format(balanced_accuracy_score(y_test_seek, y_pred_rusboost_seek),
              geometric_mean_score(y_test_seek, y_pred_rusboost_seek)))
cm_rusboost_seek = confusion_matrix(y_test_seek, y_pred_rusboost_seek)
fig, ax = plt.subplots(ncols=2)
plot_confusion_matrix(cm_rusboost_seek, classes=np.unique(dataset_seek.target),
                      ax=ax[1], title='RUSBoost classifier_seek')

plt.show()
#
# X_seek = dataset.drop(columns=['target','Estrone (ng/ml)', 'Estradiol (ng/ml)', 'DHT (ng/ml)', 'Cortisone (ng/ml)', '21-Deoxycortisol  (ng/ml)'])
# print(X)
# y = dataset['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
writer = ExcelWriter('df_created_by_python.xlsx')
dataset.to_excel(writer,'Sheet1',index=False)
writer.save()
