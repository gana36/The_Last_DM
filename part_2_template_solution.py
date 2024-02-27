# Add your imports here.
# Note: only sklearn, numpy, utils and new_utils are allowed.

import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from sklearn.metrics import accuracy_score, confusion_matrix


import utils as u
import new_utils as nu


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
        answer = {}
        
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        
        answer['nb_classes_train'] = len(np.unique(ytrain))
        answer['nb_classes_test'] = len(np.unique(ytest))
        answer['class_count_train'] = np.bincount(ytrain)
        answer['class_count_test'] = np.bincount(ytest)
        answer['length_Xtrain'] = len(Xtrain)
        answer['length_Xtest'] = len(Xtest)
        answer['length_ytrain'] = len(ytrain)
        answer['length_ytest'] = len(ytest)
        answer['max_Xtrain'] = np.max(Xtrain)
        answer['max_Xtest'] = np.max(Xtest)
        
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

        #Xtrain = Xtest = np.zeros([1, 1], dtype="float")
        #ytrain = ytest = np.zeros([1], dtype="int")

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
        
        answer = {}   
        
        def partB_sub(X, y, Xtest, ytest):
    
            clf = DecisionTreeClassifier(random_state=self.seed)
            cv = KFold(n_splits=5, random_state=self.seed,shuffle=True)
            scores = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf, cv=cv)
            results_dict_C = {}

            mean_accuracy = scores["test_score"].mean()
            std_accuracy = scores["test_score"].std()
            mean_fit_time = scores["fit_time"].mean()
            std_fit_time = scores["fit_time"].std()
        
            results_dict_C = {
             "mean_accuracy":mean_accuracy,
             "std_accuracy":std_accuracy,
             "mean_fit_time":mean_fit_time,
             "std_fit_time":std_fit_time            
                }
            
            answer_C = {}
            answer_C["scores_C"] = results_dict_C
            answer_C["clf"] = clf  
            answer_C["cv"] = cv  

            
            # Decision Tree 
            clf_D = DecisionTreeClassifier(random_state=self.seed)
            cv_D = ShuffleSplit(n_splits=5, random_state=self.seed)
            scores_D = u.train_simple_classifier_with_cv(Xtrain=X, ytrain=y, clf=clf_D, cv=cv_D)
            results_dict_D = {}
        
            mean_accuracy_D = scores_D["test_score"].mean()
            std_accuracy_D = scores_D["test_score"].std()
            mean_fit_time_D = scores_D["fit_time"].mean()
            std_fit_time_D = scores_D["fit_time"].std()

            results_dict_D = {
                 "mean_accuracy":mean_accuracy_D,
                 "std_accuracy":std_accuracy_D,
                 "mean_fit_time":mean_fit_time_D,
                 "std_fit_time":std_fit_time_D          
            }

            answer_D = {}
            answer_D["scores_D"] = results_dict_D
            answer_D["clf"] = clf_D  
            answer_D["cv"] = cv_D  

            
            
            # Logistic regression

            """ The partF dictionary should be as follows:
            partF = {
            "scores_train_F": scores_train_F,
            "scores_test_F": scores_test_F,
            "mean_cv_accuracy_F": mean_cv_accuracy_F,
            "clf": clf,
            "cv": cv,
            "conf_mat_train": conf_mat_train,
            "conf_mat_test": conf_mat_test,
            }"""

            lf_LR = LogisticRegression(max_iter=300, random_state=self.seed)
            cv_LR = ShuffleSplit(n_splits=5, random_state=self.seed)
            scores_LR = cross_validate(lf_LR,X, y, cv=cv_LR, return_train_score=True)
            results_dict_F = {}
 
            mean_cv_accuracy_F = scores_LR["test_score"].mean()

            lf_LR.fit(X, y)
    
            scores_train_F = lf_LR.score(X,y)
            scores_test_F = lf_LR.score(Xtest,ytest)
            
            conf_mat_train = confusion_matrix(y, lf_LR.predict(X))
            conf_mat_test = confusion_matrix(ytest, lf_LR.predict(Xtest))

            results_dict_F = {
            "scores_train_F": scores_train_F,
            "scores_test_F": scores_test_F,
            "mean_cv_accuracy_F": mean_cv_accuracy_F,
            "clf": lf_LR,
            "cv": cv_LR,
            "conf_mat_train": conf_mat_train,
            "conf_mat_test": conf_mat_test,
            }
         
            
            # Calculate class counts for training and testing sets
            class_count_train = list(np.bincount(y))
            class_count_test = list(np.bincount(ytest))
    
            answer = {}
            answer["partC"] = answer_C
            answer["partD"] = answer_D
            answer["partF"] = results_dict_F
            answer["ntrain"] = len(y)
            answer["ntest"] = len(ytest)
            answer["class_count_train"] = class_count_train
            answer["class_count_test"] = class_count_test
            return answer
                

        for ntr, nte in zip(ntrain_list, ntest_list):
            X_r = X[0:ntr, :]
            y_r = y[0:ntr]
            Xtest_r = Xtest[0:nte, :]
            ytest_r = ytest[0:nte]
            answer[ntr] = partB_sub(X_r, y_r, Xtest_r, ytest_r)

        return answer