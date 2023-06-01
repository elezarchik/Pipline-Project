import hydra
import os
import numpy as np
from omegaconf import OmegaConf, DictConfig
import logging
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score as PS
from sklearn.metrics import recall_score as RS
from sklearn.metrics import f1_score as FS
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import classification_report as CR
from hydra.utils import get_original_cwd, to_absolute_path
from sklearn.model_selection import train_test_split
from  matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
import sklearn as sl
from sklearn.tree import export_graphviz
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB as GNB

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config_ml')
def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Загружаем данные
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/Y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/Y_test.npy')

    params = OmegaConf.to_container(cfg['params'])
    model_name = params['model']
    if model_name == 'LogicticRegression':
        model = LogisticRegression()
        par = {'max_iter': params['max_iter'], 'penalty': params['penalty'],'tol': params['tol']}
        model.set_params(**par)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        log.info([model_name,classification_report(y_test, y_pred)])

    if model_name == 'Dtree':
        dt = DT(random_state=0)
        par = {'max_depth': range(params['min_depth'], params['max_depth'], params['depth_step']), 'min_samples_split': range(params['min_min_samples_split'], params['max_min_samples_split'], params['split_step']), 'min_samples_leaf': range(params['min_min_samples_leaf'], params['max_min_samples_leaf'], params['leaf_step'])}
        search = GridSearchCV(dt, par, n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        log.info([model_name, search.best_params_,classification_report(y_test, y_pred)])
        print([search.best_params_, classification_report(y_test, y_pred)])

    if model_name == 'KNN':
        dt = KN(n_jobs=-1)
        par = {'n_neighbors': range(params['min_n_neighbors'], params['max_n_neighbors']), 'leaf_size': range(params['leaf_size_min'], params['leaf_size_max'], params['leaf_size_step'])}
        search = GridSearchCV(dt, par, n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        log.info([model_name, search.best_params_,classification_report(y_test, y_pred)])
        print(classification_report(y_test, y_pred))

    if model_name == 'SVM':
        dt = SVC()
        par = {'C': range(params['C_min'], params['C_max']), 'degree': range(params['degree_min'], params['degree_max'])}
        search = GridSearchCV(dt, par, n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        log.info([model_name, search.best_params_,classification_report(y_test, y_pred)])
        print(classification_report(y_test, y_pred))

    if model_name == 'RF':
        dt = RandomForestClassifier()
        par = {'max_depth': range(params['max_depth_min'],params['max_depth_max'], params['max_depth_step']), 'min_samples_split': range(params['min_min_samples_split'], params['max_min_samples_split'], params['split_step']), 'min_samples_leaf': range(params['min_min_samples_leaf'], params['max_min_samples_leaf'], params['leaf_step'])}
        search = GridSearchCV(dt, par, n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        log.info([model_name, search.best_params_,classification_report(y_test, y_pred)])
        print(classification_report(y_test, y_pred))
    if model_name == 'GBC':
        dt = GradientBoostingClassifier()
        par = {'min_samples_split':range(params['min_samples_split'],params['max_samples_split'], params['step_samples_split']), 'min_samples_leaf':range(params['min_samples_leaf'],params['max_samples_leaf'], params['step_samples_leaf']), 'max_depth':range(params['min_depth'],params['max_depth'], params['step_depth'])}
        search = GridSearchCV(dt, par, n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        log.info([model_name, search.best_params_, classification_report(y_test, y_pred)])
        print(classification_report(y_test, y_pred))
if __name__ == "__main__":
    my_app()