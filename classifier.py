from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
import math
import warnings
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from self_paced_ensemble import SelfPacedEnsembleClassifier
from sklearn.tree import DecisionTreeClassifier


def SVC_F(X_train, y_train, X_test):
    NBM = [SVC(kernel='linear', C=50),  # 其中C即为上文所提到的松弛变量，C越小间隔范围越大
           SVC(kernel='rbf', C=50, gamma=1.),  # gamma为高斯核的带宽
           SVC(kernel='poly', C=60, degree=3)]  # degree为多项式次数
    NAME = ["LINEAR", "RBF", "poly"]
    Y = []
    for itr, itrname in zip(NBM, NAME):
        # 训练过程
        # print("Training...")
        itr.fit(X_train, y_train.astype("int"))
        # print("Applying...")
        y_pdt = itr.predict(X_test)
        Y.append(y_pdt)
    return Y


def KNN(X_train, y_train, X_test):
    # 定义随KNN分类器  参数N为邻居数目
    knn = KNeighborsClassifier(n_neighbors=10)
    # 训练过程  X为特征向量，y为标签数据/向量
    knn.fit(X_train, y_train.astype('int'))
    y_pdt = knn.predict(X_test)
    #print(y_pdt)

    return y_pdt

def SPE_classifier(X_train, y_train, X_test):
    clf = SelfPacedEnsembleClassifier(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=10,)
    clf.fit(X_train, y_train)
    # Predict with an SPE classifier
    z = clf.predict(X_test)
    return z

# Logistic Regression Classifier
def logistic_regression_classifier(X_train, y_train, X_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pdt = model.predict(X_test)
    return y_pdt

# Random Forest Classifier
def random_forest_classifier(X_train, y_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(X_train, y_train.astype('int'))
    y_pdt = model.predict(X_test)
    return y_pdt

# Decision Tree Classifier
def decision_tree_classifier(X_train, y_train, X_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train.astype('int'))
    y_pdt = model.predict(X_test)
    return y_pdt


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(X_train, y_train, X_test):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train.astype('int'))
    y_pdt = model.predict(X_test)
    return y_pdt

def bys(X_train, y_train, X_test):
    # 定义高斯分类器类
    gnb = GaussianNB()
    # 训练过程
    gnb.fit(X_train, y_train)
    bye_pdt = gnb.predict(X_test)
    return bye_pdt


def classif(F1, label, F1_test, test, trainnew, Label):
    # print('--------------------------------SPE分类器----------------------------------------')
    # SPE_pdt = SPE_classifier(F1, label, F1_test)
    # SPE_YL = np.mat(SPE_pdt).T  # 预测标签
    # SPE_TL = np.mat(np.array(test[:, 0])).T  # 测试集标签
    # PRindex(SPE_YL, SPE_TL, test)
    # print('--------------------------------SPE原始训练集----------------------------------------')
    # SPE_pdt1 = SPE_classifier(trainnew, Label, F1_test)
    # SPE_YL1 = np.mat(SPE_pdt1).T
    # SPE_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(SPE_YL1, SPE_TL1, test)
    # print('--------------------------------SPE分类器----------------------------------------')


    print('--------------------------------SVM分类器----------------------------------------')
    SVM_pdt = SVC_F(F1, label, F1_test)
    SVM_YL = np.mat(SVM_pdt[0]).T  # 预测标签
    SVM_TL = np.mat(np.array(test[:, 0])).T  # 测试集标签
    PRindex(SVM_YL, SVM_TL, test)
    # print('--------------------------------SVM原始训练集----------------------------------------')
    # SVM_pdt1 = SVC_F(trainnew, Label, F1_test)
    # SVM_YL1 = np.mat(SVM_pdt1[0]).T
    # SVM_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(SVM_YL1, SVM_TL1, test)
    # print('--------------------------------SVM1分类器----------------------------------------')
    # SVM1_pdt = SVC_F(F1, label, F1_test)
    # SVM1_YL = np.mat(SVM1_pdt[1]).T
    # SVM1_TL = np.mat(np.array(test[:, 0])).T
    # PRindex(SVM1_YL, SVM1_TL, test)
    # print('--------------------------------SVM1原始训练集----------------------------------------')
    # SVM_pdt2 = SVC_F(trainnew, Label, F1_test)
    # SVM_YL2 = np.mat(SVM_pdt2[1]).T
    # SVM_TL2 = np.mat(np.array(test[:, 0])).T
    # PRindex(SVM_YL2, SVM_TL2, test)
    # print('--------------------------------SVM2分类器----------------------------------------')
    # SVM2_pdt = SVC_F(F1, label, F1_test)
    # SVM2_YL = np.mat(SVM2_pdt[2]).T
    # SVM2_TL = np.mat(np.array(test[:, 0])).T
    # PRindex(SVM2_YL, SVM2_TL, test)
    # print('--------------------------------SVM2原始训练集----------------------------------------')
    # SVM_pdt3 = SVC_F(trainnew, Label, F1_test)
    # SVM_YL3 = np.mat(SVM_pdt3[2]).T
    # SVM_TL3 = np.mat(np.array(test[:, 0])).T
    # PRindex(SVM_YL3, SVM_TL3, test)
    # print('--------------------------------SVM分类器----------------------------------------')

    print('--------------------------------KNN分类器----------------------------------------')
    KNN_pdt = KNN(F1, label, F1_test)
    KNN_YL = np.mat(KNN_pdt).T
    KNN_TL = np.mat(np.array(test[:, 0])).T
    PRindex(KNN_YL, KNN_TL, test)
    # print('--------------------------------KNN原始训练集----------------------------------------')
    # KNN_pdt1 = KNN(trainnew, Label, F1_test)
    # KNN_YL1 = np.mat(KNN_pdt1).T
    # KNN_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(KNN_YL1, KNN_TL1, test)
    # print('--------------------------------KNN分类器----------------------------------------')

    # print('--------------------------------GBDT分类器----------------------------------------')
    # Mul_pdt = gradient_boosting_classifier(F1, label, F1_test)
    # MUL_YL = np.mat(Mul_pdt).T
    # Mul_TL = np.mat(np.array(test[:, 0])).T
    # PRindex(MUL_YL, Mul_TL, test)
    # print('--------------------------------GBDT分类器----------------------------------------')
    # print('--------------------------------GBDT原始训练集----------------------------------------')
    # Mul_pdt1 = gradient_boosting_classifier(trainnew, Label, F1_test)
    # MUL_YL1 = np.mat(Mul_pdt1).T
    # Mul_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(MUL_YL1, Mul_TL1, test)
    # # print('--------------------------------GBDT原始训练集----------------------------------------')
    #
    print('--------------------------------决策树分类器----------------------------------------')
    Tree_pdt = decision_tree_classifier(F1, label, F1_test)
    Tree_YL = np.mat(Tree_pdt).T
    Tree_TL = np.mat(np.array(test[:, 0])).T
    PRindex(Tree_YL, Tree_TL, test)
    # print('--------------------------------决策树分类器----------------------------------------')
    # print('--------------------------------决策树原始训练集----------------------------------------')
    # Tree_pdt1 = decision_tree_classifier(trainnew, Label, F1_test)
    # Tree_YL1 = np.mat(Tree_pdt1).T
    # Tree_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(Tree_YL1, Tree_TL1, test)
    # print('--------------------------------决策树原始训练集----------------------------------------')

    # print('--------------------------------logistic_regression分类器----------------------------------------')
    # Log_pdt = logistic_regression_classifier(F1, label, F1_test)
    # Log_YL = np.mat(Log_pdt).T
    # Log_TL = np.mat(np.array(test[:, 0])).T
    # PRindex(Log_YL, Log_TL, test)
    # print( '--------------------------------logistic_regression分类器----------------------------------------')
    # print('--------------------------------logistic_regression分类器原始训练集----------------------------------------')
    # Log_pdt1 = logistic_regression_classifier(trainnew, Label, F1_test)
    # Log_YL1 = np.mat(Log_pdt1).T
    # Log_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(Log_YL1, Log_TL1, test)
    # print('--------------------------------logistic_regression分类器原始训练集----------------------------------------')

    # print('--------------------------------随机森林分类器----------------------------------------')
    # bys_pdt = random_forest_classifier(F1, label, F1_test)
    # bys_YL = np.mat(bys_pdt).T
    # bys_TL = np.mat(np.array(test[:, 0])).T
    # PRindex(bys_YL, bys_TL, test)
    # print('--------------------------------随机森林原始训练集----------------------------------------')
    # bys_pdt1 = random_forest_classifier(trainnew, Label, F1_test)
    # bys_YL1 = np.mat(bys_pdt1).T
    # bys_TL1 = np.mat(np.array(test[:, 0])).T
    # PRindex(bys_YL1, bys_TL1, test)
    # print('--------------------------------随机森林分类器----------------------------------------')

def PRindex(M_YL, M_TL, test):
    errArr = np.mat(np.ones((len(test), 1)))
    TP = errArr[(M_YL == 1) & (M_YL == M_TL)].sum()
    TN = errArr[(M_YL == 0) & (M_YL == M_TL)].sum()
    FP = errArr[(M_YL == 1) & (M_YL != M_TL)].sum()
    FN = errArr[(M_YL == 0) & (M_YL != M_TL)].sum()


    Specificity_Test = errArr[(M_YL == 0) & (M_TL == 0)].sum() / (
            errArr[(M_YL == 0) & (M_TL == 0)].sum() + FP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("TP: ", TP)
        print("TN: ", TN)
        print("FN: ", FN)
        print("FP: ", FP)
        print("accuracy：", accuracy_score(M_TL, M_YL))
        print("recall: ", recall_score(M_TL, M_YL))
        print("precision: ", precision_score(M_TL, M_YL))
        print("f1_score: ", f1_score(M_TL, M_YL))
        print("f2_score: ", (5 * precision_score(M_TL, M_YL) * recall_score(M_TL, M_YL)) /
              (4 * precision_score(M_TL, M_YL) + recall_score(M_TL, M_YL)))
        print("G-mean：", math.sqrt(recall_score(M_TL, M_YL) * Specificity_Test))
        print("AUC: ", roc_auc_score(M_TL, M_YL))
        print("TC:", FN * ((TN + FP) / (TP + FN)) + FP)
