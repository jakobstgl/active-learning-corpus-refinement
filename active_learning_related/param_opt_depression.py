import torch
import pandas as pd 
import numpy as np
import pickle
from datasets import Dataset, DatasetDict
from typing import Dict, Any, Union
from sklearn.metrics import f1_score, accuracy_score
from optuna import Trial
from active_learning_related.modules.smalltext_pipeline import SmallTextPipeline
from setfit import SetFitModel, Trainer, TrainingArguments
import gc
import optuna
import json

def compute_metrics(y_pred, y_true):
    from sklearn.metrics import f1_score
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "f1": f1_score(y_true, y_pred, pos_label=0),
        "accuracy": accuracy_score(y_true, y_pred)
    }

def objective(metrics):
    print(metrics)
    return metrics['f1']

depression_pipeline = SmallTextPipeline.load_learning_state("")

with open('../data/small_text_datasets.pkl', 'rb') as f:
    small_text_datasets = pickle.load(f)

train_dataset = small_text_datasets['psyC']['smalltext_train_dset']
test_dataset = small_text_datasets['psyC']['smalltext_test_dset']

# Update train dataset with labels of other pipeline

train_dataset.y[depression_pipeline.history[2]['indices_labeled']] = depression_pipeline.history[2]['indices_labels']

indices_initial = depression_pipeline.history[2]['indices_labeled']


train = Dataset.from_dict({
    'text': train_dataset.x,
    'label': train_dataset.y
})

test = Dataset.from_dict({
    'text': test_dataset.x,
    'label': test_dataset.y
})

train_labeled = train.filter(lambda x: x['label'] != -1)

def model_init(params: Dict[str, Any]) -> SetFitModel:
    params = params or {}
    
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", **params, use_differentiable_head=True)

    # Move model body to GPU if available
    if torch.cuda.is_available():
        model.model_body.to("cuda")

    return model

study_name = "1605_depression_hpo"
storage_name = "sqlite:////home/data/optuna_logs/{}.db".format(study_name)

def objective(trial):
    # Sample hyperparameters
    body_learning_rate = trial.suggest_float("body_learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_iterations = trial.suggest_int("num_iterations", 10, 50, step=10)
    max_steps = trial.suggest_int("max_steps", 20, 300, step=20)
    
    training_args = TrainingArguments(
        body_learning_rate = body_learning_rate,
        batch_size = batch_size,
        #num_iterations = num_iterations,
        max_steps = max_steps,
        seed=1234,
        use_amp=True
    )
    # Reinitialize model each trial
    model = model_init(params= {})
    
    trainer = Trainer(
        model=model,
        train_dataset=train_labeled,
        eval_dataset=test,
        metric=compute_metrics,
        args = training_args
    )

    # Training (currently only model body)
    trainer.train()
    #trainer.train_classifier()

    # Evaluate
    metrics = trainer.evaluate()
    print(f"Trial {trial.number} — F1-Score: {metrics['f1']}")

    # Save results
    result = {
        "trial": trial.number,
        "params": trial.params,
        "f1_score": metrics["f1"],
        "accuracy": metrics["accuracy"]
    }

    with open("trial_results_depression.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return metrics["f1"]

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    sampler=optuna.samplers.TPESampler(seed=1234),
    load_if_exists=True
)
study.optimize(objective, n_trials=20)

# Save the sampler state after the study
with open("depression_sampler.pkl", "wb") as fout:
    pickle.dump(study.sampler, fout)
