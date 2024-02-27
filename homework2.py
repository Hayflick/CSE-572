import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn import preprocessing

from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC

plt.style.use('ggplot')

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


dataset = pd.read_csv('train.csv')
dataset.head()

dataset.describe(include="all")

dataset.shape

dataset.isnull().sum(axis=0)

sns.countplot(dataset['Embarked'])

# File missing values in embarked with S which is the most frequent item.
dataset = dataset.fillna({"Embarked": "S"})

## One hot encoding is used since no ordering is available for Sex (male, female) feature.
dataset = pd.get_dummies(dataset, columns=['Sex'])
dataset.head()

## One hot encoding is used since no ordering is available for Sex (male, female) feature.
dataset = pd.get_dummies(dataset, columns=['Embarked'])


feat_names = ['Pclass', 'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Parch', 'SibSp', 'Fare']
targ_names = ['Dead (0)', 'Survived (1)']  # 0 - Dead, 1 - Survived

train_class = dataset[['Survived']]
train_feature = dataset[feat_names]

clf = DecisionTreeClassifier(random_state=0)
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
scores = cross_validate(clf, train_feature, train_class, scoring=scoring)
# print(scores.keys())

print('Accuracy score : %.3f' % scores['test_acc'].mean())
print('Precision score : %.3f' % scores['test_prec_macro'].mean())
print('Recall score : %.3f' % scores['test_rec_macro'].mean())
print('F1 score : %.3f' % scores['test_f1_macro'].mean())

para_grid = {
    'min_samples_split': range(10, 500, 20),
    'max_depth': range(1, 20, 2),
    'criterion': ("gini", "entropy")
}

clf_tree = DecisionTreeClassifier()
clf_cv = GridSearchCV(clf_tree,
                      para_grid,
                      scoring='accuracy',
                      cv=5,
                      n_jobs=-1)
clf_cv.fit(train_feature, train_class)

best_parameters = clf_cv.best_params_
print(best_parameters)

clf = clf_cv.best_estimator_
scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
scores = cross_validate(clf, train_feature, train_class, scoring=scoring)
# print(scores.keys())

print('Accuracy score : %.3f' % scores['test_acc'].mean())
print('Precision score : %.3f' % scores['test_prec_macro'].mean())
print('Recall score : %.3f' % scores['test_rec_macro'].mean())
print('F1 score score : %.3f' % scores['test_f1_macro'].mean())

# Create a holdout sample for further testing
# train_class, train_feature
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_class, test_size=0.33)
print(str(X_train.shape) + "," + str(y_train.shape))
print(str(X_test.shape) + "," + str(y_test.shape))

clf2 = clf_cv.best_estimator_
clf2.fit(X_train, y_train)
predictions = clf2.predict(X_test)
print(metrics.classification_report(y_test, predictions, target_names=targ_names, digits=3))

fig, ax = plt.subplots(figsize=(7, 3))
visualizer = ClassificationReport(clf2, classes=targ_names, support=True, cmap='RdPu')
visualizer.score(X_test, y_test)
for label in visualizer.ax.texts:
    label.set_size(14)
g = visualizer.poof()

fig, ax = plt.subplots(figsize=(3, 3))
cm = ConfusionMatrix(clf2, classes=[0, 1], cmap='RdPu')
cm.score(X_test, y_test)
for label in cm.ax.texts:
    label.set_size(14)
cm.poof()

modelviz = clf_cv.best_estimator_
visualizer = ROCAUC(modelviz, classes=["Dead", "Survived"])

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()  # Finalize and render the figure

import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

data = export_graphviz(clf, out_file=None, feature_names=feat_names, class_names=targ_names,
                       filled=True, rounded=True,
                       special_characters=True)
graph = graphviz.Source(data)
graph.render(format='png', cleanup=True)

importances = clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



all = pd.concat([train, test], sort = False)

all['Age'] = all['Age'].fillna(value=all['Age'].median())
all['Fare'] = all['Fare'].fillna(value=all['Fare'].median())



all['Embarked'] = all['Embarked'].fillna('S')

all.loc[ all['Age'] <= 16, 'Age'] = 0
all.loc[(all['Age'] > 16) & (all['Age'] <= 32), 'Age'] = 1
all.loc[(all['Age'] > 32) & (all['Age'] <= 48), 'Age'] = 2
all.loc[(all['Age'] > 48) & (all['Age'] <= 64), 'Age'] = 3
all.loc[ all['Age'] > 64, 'Age'] = 4

import re


def get_title(name):
    title_search = re.search(' ([A-Za-z]+\.)', name)

    if title_search:
        return title_search.group(1)
    return ""



all['Title'] = all['Name'].apply(get_title)
all['Title'].value_counts()



all['Title'] = all['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')
all['Title'] = all['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')
all['Title'] = all['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')
all['Title'] = all['Title'].replace(['Mme.'], 'Mrs.')



#Cabin
all['Cabin'] = all['Cabin'].fillna('Missing')
all['Cabin'] = all['Cabin'].str[0]



#Family Size & Alone
all['Family_Size'] = all['SibSp'] + all['Parch'] + 1
all['IsAlone'] = 0
all.loc[all['Family_Size']==1, 'IsAlone'] = 1



#Drop unwanted variables
all_1 = all.drop(['Name', 'Ticket'], axis = 1)



all_dummies = pd.get_dummies(all_1)

all_train = all_dummies[all_dummies['Survived'].notna()]

all_test = all_dummies[all_dummies['Survived'].isna()]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_train.drop(['PassengerId','Survived'],axis=1),
                                                    all_train['Survived'], test_size=0.30,
                                                    random_state=101)



RF_Model = RandomForestClassifier()



RF_Model = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=7, max_features='sqrt',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=6,
                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)



RF_Model.fit(X_train, y_train)



predictions = RF_Model.predict(X_test)

print(np.mean(cross_val_score(clf, X_train, y_train, cv=5)))
