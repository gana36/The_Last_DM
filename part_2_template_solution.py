# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils as u
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # ---------------------------------------------------------

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all 10 classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
    """

    def partA(
        self,
    ) -> tuple[
        dict[str, Any],
        NDArray[np.floating],
        NDArray[np.int32],
        NDArray[np.floating],
        NDArray[np.int32],
    ]:
        X, y, Xtest, ytest = u.prepare_data()
        
        answer = {}
        answer["nb_classes_train"] = len(np.unique(y))
        answer["nb_classes_test"] = len(np.unique(ytest))
        answer["class_count_train"] = np.bincount(y)
        answer["class_count_test"] = np.bincount(ytest)
        answer["length_Xtrain"] = X.shape[0]
        answer["length_Xtest"] = Xtest.shape[0]
        answer["length_ytrain"] = len(y)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(X)
        answer["max_Xtest"] = np.max(Xtest)
        
        # Enter your code and fill the `answer`` dictionary

        # `answer` is a dictionary with the following keys:
        # - nb_classes_train: number of classes in the training set
        # - nb_classes_test: number of classes in the testing set
        # - class_count_train: number of elements in each class in the training set
        # - class_count_test: number of elements in each class in the testing set
        # - length_Xtrain: number of elements in the training set
        # - length_Xtest: number of elements in the testing set
        # - length_ytrain: number of labels in the training set
        # - length_ytest: number of labels in the testing set
        # - max_Xtrain: maximum value in the training set
        # - max_Xtest: maximum value in the testing set

        # return values:
        # Xtrain, ytrain, Xtest, ytest: the data used to fill the `answer`` dictionary

        """Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        ytrain = ytest = np.zeros([1], dtype="int")"""
        Xtrain = X
        ytrain = y
        #print('Question A')
        #print(answer)
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    B.  Repeat part 1.C, 1.D, and 1.F, for the multiclass problem. 
        Use the Logistic Regression for part F with 300 iterations. 
        Explain how multi-class logistic regression works (inherent, 
        one-vs-one, one-vs-the-rest, etc.).
        Repeat the experiment for ntrain=1000, 5000, 10000, ntest = 200, 1000, 2000.
        Comment on the results. Is the accuracy higher for the training or testing set?
        What is the scores as a function of ntrain.

        Given X, y from mnist, use:
        Xtrain = X[0:ntrain, :]
        ytrain = y[0:ntrain]
        Xtest = X[ntrain:ntrain+test]
        ytest = y[ntrain:ntrain+test]
    """

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
        ntest_list: list[int] = [],
    ) -> dict[int, dict[str, Any]]:
        """ """
        answer = {}
        train_list = [1000,5000,10000]
        test_list = [ 200, 1000, 2000]
        for i in range(0,len(train_list)):
            train_val = train_list[i]
            test_val= test_list[i]
            X_train = X[:train_val]
            y_train = y[:train_val]
            X_test = Xtest[:test_val]
            y_test = ytest[:test_val]
            #answer
            ##Part C 
            partC ={}
            scores = u.train_simple_classifier_with_cv(Xtrain=X_train,ytrain=y_train,clf=DecisionTreeClassifier(random_state=self.seed),cv=KFold(n_splits=5))
            score_dict={}
            for key,array in scores.items():
                if(key=='fit_time'):
                    score_dict['mean_fit_time'] = array.mean()
                    score_dict['std_fit_time'] = array.std()
                if(key=='test_score'):
                    score_dict['mean_accuracy'] = array.mean()
                    score_dict['std_accuracy'] = array.std()
            partC["clf"] = DecisionTreeClassifier(random_state=self.seed)  # the estimator (classifier instance)
            partC["cv"] = KFold(n_splits=5)
            partC["scores"] = score_dict

            ##Part D
            partD = {}
            scores_dt= u.train_simple_classifier_with_cv(Xtrain=X_train,ytrain=y_train,clf=DecisionTreeClassifier(random_state=self.seed),cv=ShuffleSplit(n_splits=5,random_state=self.seed))
            partD["mean_fit_time"] = np.mean(scores_dt['fit_time'])
            partD["std_fit_time"] = np.std(scores_dt['fit_time'])
            partD["mean_accuracy"] = np.mean(scores_dt['test_score'])    
            partD["std_accuracy"] = np.std(scores_dt['test_score']) 
            
            partD["clf"] = DecisionTreeClassifier(random_state=self.seed)  # the estimator (classifier instance)
            partD["cv"] = ShuffleSplit(n_splits=5,random_state=self.seed)
            partD["scores"] = score_dict

            ##Part F
            partF = {}
            scores_rf = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=LogisticRegression(random_state=self.seed,max_iter=300), cv=ShuffleSplit(n_splits=5,random_state=self.seed))
            #scores_dt = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=DecisionTreeClassifier(random_state=self.seed), cv=ShuffleSplit(n_splits=5,random_state=self.seed)) 
            scores_RF={}
            scores_DT={}
            scores_RF["mean_fit_time"] = np.mean(scores_rf['fit_time'])
            scores_RF["std_fit_time"] = np.std(scores_rf['fit_time'])
            scores_RF["mean_accuracy"] = np.mean(scores_rf['test_score'])    
            scores_RF["std_accuracy"] = np.std(scores_rf['test_score'])    
            scores_RF["model_highest_accuracy"] = np.max(scores_rf['test_score'])
            scores_RF["model_lowest_variance"] = np.min(np.var(scores_rf['test_score']))
            scores_RF["model_fastest"] = np.min(scores_rf['fit_time']) 
            scores_DT["mean_fit_time"] = np.mean(scores_dt['fit_time'])
            scores_DT["std_fit_time"] = np.std(scores_dt['fit_time'])
            scores_DT["mean_accuracy"] = np.mean(scores_dt['test_score'])    
            scores_DT["std_accuracy"] = np.std(scores_dt['test_score'])  
            scores_DT["model_highest_accuracy"] = np.max(scores_dt['test_score'])
            scores_DT["model_lowest_variance"] = np.min(np.var(scores_dt['test_score']))
            scores_DT["model_fastest"] = np.min(scores_dt['fit_time'])  
            partF["clf_RF"] =  RandomForestClassifier(random_state=self.seed)
            partF["clf_DT"] = DecisionTreeClassifier(random_state=self.seed)
            partF["cv"] = ShuffleSplit(n_splits=5,random_state=self.seed)
            partF["scores_RF"] = scores_RF
            partF["scores_DT"] = scores_DT
            answer[train_list[i]] = {}
            answer[train_list[i]]["partC"] = partC
            answer[train_list[i]]["partD"] = partD
            answer[train_list[i]]["partF"] = partF
            answer[train_list[i]]["ntrain"] = train_list[i]
            answer[train_list[i]]["ntest"] = test_list[i]
            answer[train_list[i]]["class_count_train"] = list(np.bincount(y))
            answer[train_list[i]]["class_count_test"] = list(np.bincount(ytest))
        # Enter your code and fill the `answer`` dictionary
       

        """
        `answer` is a dictionary with the following keys:
           - 1000, 5000, 10000: each key is the number of training samples

           answer[k] is itself a dictionary with the following keys
            - "partC": dictionary returned by partC section 1
            - "partD": dictionary returned by partD section 1
            - "partF": dictionary returned by partF section 1
            - "ntrain": number of training samples
            - "ntest": number of test samples
            - "class_count_train": number of elements in each class in
                               the training set (a list, not a numpy array)
            - "class_count_test": number of elements in each class in
                               the training set (a list, not a numpy array)
        """
        #print('Question B')
        #print(answer)
        return answer
