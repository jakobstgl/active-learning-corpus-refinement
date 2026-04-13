import torch
import pandas as pd 
from sklearn.metrics import (
    accuracy_score,
    f1_score
)
from small_text import (
    PoolBasedActiveLearner,
    SubsamplingQueryStrategy,
    SetFitClassificationFactory,
    SetFitModelArguments,
)
import numpy as np
import random
import gc


def smalltext_config(train_data, num_classes: int, model_name: str, strategy, subsampling: bool, subsampling_size: int, classification_kwargs: dict, model_args: dict):

    # Subsamplingquerystrategy or not
    if subsampling is True:
        query_strategy = SubsamplingQueryStrategy(strategy, subsample_size = subsampling_size)

    elif subsampling is False:
        query_strategy = strategy

    # Classifier
    setfit_model_args = SetFitModelArguments(model_name, **model_args)
    
    # Classification Factory
    clf_factory = SetFitClassificationFactory(
        setfit_model_args,
        num_classes,
        classification_kwargs   
    )

    # Active Learner
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_data)

    return clf_factory, active_learner


    
    # does the initialization steps
    # saves a initialized learner state
    # arguments are everything from small-text


class SmallTextPipeline: #create pipeline for a certain disease
    
    def __init__(self, train, test, disease: str, clf_factory, active_learner):
        self.train = train
        self.test = test
        self.stopping_subset = None
        self.stopping_criterion = None
        self.indices_labeled = None
        self.clf_factory = clf_factory
        self.active_learner = active_learner
        self.history = []

    def initialize_learner(self, indices_initial, save_to_disk: bool = False, path = None):

        if not isinstance(indices_initial, np.ndarray):
            indices_initial= np.array(indices_initial)

        self.indices_labeled = indices_initial
        self.active_learner.initialize(indices_initial)
        y_pred, y_pred_test, train_acc, test_acc, test_fscore = self.evaluate() # should return all evaluated metrics
        self.log_metrics(
            num_labeled = len(self.active_learner.y),
            indices_labeled = self.active_learner.indices_labeled,
            indices_labels = self.active_learner.y,
            y_pred_train = y_pred,
            y_pred_test = y_pred_test,
            train_acc = train_acc,
            test_acc = test_acc,
            test_fscore = test_fscore               
        )

        if save_to_disk is True:
            self.save_learning_state(path)


    def create_stopping_subset(self, size = 50000):
        indices = random.sample(population=range(len(self.train)), k=size)
        self.stopping_subset = self.train[indices]
    
    def check_stopping(self, stopping_criterion, prediction_subset: bool = True, subset_size: int = None) -> None:
        
        # Save Stopping Criterion
        if self.stopping_criterion is None:
            self.stopping_criterion = stopping_criterion

        subset = None
    
        if prediction_subset is False: 
            subset = self.train
        elif self.stopping_subset is None:
            self.create_stopping_subset(size=subset_size)
            subset = self.stopping_subset
        else:
            subset = self.stopping_subset
            
        response = stopping_criterion.stop(
            predictions = self.active_learner.classifier.predict(subset),
        )
        print(f'Stop: {response}')
        return response

    def evaluate(self):
        self.train.y[self.indices_labeled] = self.active_learner.y #Change labeled indices in training dataset
    
        y_pred = self.active_learner.classifier.predict(self.train[self.indices_labeled]) #generate predictions from classifier on the train set
        y_pred_test = self.active_learner.classifier.predict(self.test) #same for test-set

        train_acc = accuracy_score(y_true = self.train.y[self.indices_labeled], y_pred=y_pred) #compute accuracy for training set
        test_acc = accuracy_score(y_true = self.test.y, y_pred=y_pred_test)
        test_fscore = f1_score(y_true = self.test.y, y_pred = y_pred_test, pos_label = 0) #pos_label maybe needs fix

        
        print('Train accuracy: {:.2f}'.format(train_acc))
        print('Test accuracy: {:.2f}'.format(test_acc))
        print('Test F1-Score: {:.2f}'.format(test_fscore))

        return y_pred, y_pred_test, train_acc, test_acc, test_fscore
        
    def log_metrics(self, num_labeled, indices_labeled, indices_labels, y_pred_train, y_pred_test, train_acc, test_acc, test_fscore, stopping_response = None):
        self.history.append({
            'number_labeled': num_labeled,
            'indices_labeled': indices_labeled,
            'indices_labels': indices_labels,
            'pred_train': y_pred_train,
            'pred_test': y_pred_test,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1score': test_fscore
        })

        if stopping_response is not None:
            self.history[-1]['stopping_criterion_response'] = stopping_response

    def start_loop_async(self, num_samples: int, path_to_labels: str, path_to_disk: str=None):
        indices_queried = self.active_learner.query(num_samples)
        df = pd.DataFrame(data= {'snippet': [self.train.x[i] for i in indices_queried], 'label': ''})
        df.to_csv(path_to_labels, index = False)

        if path_to_disk is not None:
            self.save_learning_state(path_to_disk)

    def continue_loop_async(self, path_to_labels: str, path_to_disk: str=None) -> None:
        df=pd.read_csv(path_to_labels, header=1, usecols=[0,1], sep=";")

        if df['label'] is None:
            raise Exception("No labels provided")
        
        labels_stored = df['label'].astype(int)

        self.active_learner.update(np.array(labels_stored))
        
        # memory fix: https://github.com/UKPLab/sentence-transformers/issues/1793
        gc.collect()
        torch.cuda.empty_cache()

        self.indices_labeled = self.active_learner.indices_labeled

        print('---------------')
        print(f'Iteration #{1} ({len(self.indices_labeled)} samples)')
            
        y_pred, y_pred_test, train_acc, test_acc, test_fscore = self.evaluate() # should return all evaluated metrics 
        #stopping_response = self.check_stopping(self.stopping_criterion)
        self.log_metrics(
            num_labeled = len(self.active_learner.y),
            indices_labeled = self.active_learner.indices_labeled,
            indices_labels = self.active_learner.y,
            y_pred_train = y_pred,
            y_pred_test = y_pred_test,
            train_acc = train_acc,
            test_acc = test_acc,
            test_fscore = test_fscore
        )

        if path_to_disk is not None:
            self.save_learning_state(path_to_disk)



    def start_loop(self, num_queries: int, num_samples: int, save_to_disk: bool = True, path: str = None) -> None:
        for i in range(num_queries):
            labels_stored = []  

            # Query Active Learner
            indices_queried = self.active_learner.query(num_samples)
       
            # Annotate Queries
            print(f"Iteration #{i}: Please label the following samples:")
            for idx in indices_queried:
                # Print or display the sample
                print(f"Sample {idx}: {self.train.x[idx]}") 
                label = input("Enter label: ")  # Collect the label
                labels_stored.append(int(label))
            
            self.active_learner.update(np.array(labels_stored))

            # memory fix: https://github.com/UKPLab/sentence-transformers/issues/1793
            gc.collect()
            torch.cuda.empty_cache()
            
            self.indices_labeled = self.active_learner.indices_labeled
            
            print('---------------')
            print(f'Iteration #{i} ({len(self.indices_labeled)} samples)')
            
            y_pred, y_pred_test, train_acc, test_acc, test_fscore = self.evaluate() # should return all evaluated metrics 
            stopping_response = self.check_stopping(self.stopping_criterion)
            self.log_metrics(
                num_labeled = len(self.active_learner.y),
                indices_labeled = self.active_learner.indices_labeled,
                indices_labels = self.active_learner.y,
                y_pred_train = y_pred,
                y_pred_test = y_pred_test,
                train_acc = train_acc,
                test_acc = test_acc,
                test_fscore = test_fscore,
                stopping_response = stopping_response
            )

        if save_to_disk is True:
            self.save_learning_state(path)
        
        
    def save_learning_state(self, path: str):
        import dill as pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        

    @classmethod
    def load_learning_state(cls, path):
        import dill as pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
        
