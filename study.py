from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 20
DATA = Path(__file__).resolve().parent / 'data'
TASKS = ('classification', 'regression')


def load(task):
    if task == 'classification':
        data = pd.read_csv(DATA / 'wholesale.csv').drop_duplicates()
        return data.drop(columns='Channel'), data['Channel']
    if task == 'regression':
        data = pd.read_csv(DATA / 'sgemm.csv')
        runs = data.columns[data.columns.str.startswith('Run')]
        return data.drop(columns=runs), data[runs].mean(axis=1).rename('runtime_ms')
    raise ValueError(f'Unknown task: {task}')


def split(task):
    x, y = load(task)
    return train_test_split(x, y, test_size=0.2, random_state=SEED,
                            stratify=y if task == 'classification' else None)


def engineer(x, task):
    if task == 'classification':
        spending = x.drop(columns='Region')
        total = spending.sum(axis=1).replace(0, 1)
        regions = {f'region_{i}': (x['Region'] == i).astype(float) for i in (1, 2, 3)}
        shares = spending.div(total, axis=0).add_suffix('_share')
        return pd.concat([np.log1p(spending), shares, pd.DataFrame(regions, index=x.index)], axis=1)

    # Степени двойки и кратности плиток задают разные масштабы работы ядра.
    result = np.log2(x.assign(STRM=x.STRM + 1, STRN=x.STRN + 1, SA=x.SA + 1, SB=x.SB + 1))
    return result.assign(rows_per_thread=result.MWG - result.MDIMC,
                         cols_per_thread=result.NWG - result.NDIMC,
                         tile_area=result.MWG + result.NWG,
                         threads=result.MDIMC + result.NDIMC,
                         unroll_tile=result.KWG + result.KWI,
                         cache_a=x.SA * result.MWG, cache_b=x.SB * result.NWG)


def pipeline(task, model, prep):
    steps = []
    if 'features' in prep:
        steps.append(FunctionTransformer(engineer, kw_args={'task': task}))
    if 'scale' in prep:
        steps.append(StandardScaler())
    return make_pipeline(*steps, model)


def validate(task, factory, candidates):
    x_train, _, y_train, _ = split(task)
    classification = task == 'classification'
    cv = (StratifiedKFold if classification else KFold)(5, shuffle=True, random_state=SEED)
    scoring = 'f1_macro' if classification else 'neg_mean_absolute_error'
    rows = []
    for label, prep, params in candidates:
        scores = cross_val_score(pipeline(task, factory(params), prep), x_train, y_train,
                                 cv=cv, scoring=scoring, error_score='raise')
        rows.append({'experiment': label, 'prep': prep, 'params': params,
                     'cv_mean': scores.mean() * (1 if classification else -1),
                     'cv_std': scores.std()})
    return pd.DataFrame(rows).sort_values('cv_mean', ascending=not classification, kind='stable')


def evaluate(task, model, prep='raw'):
    x_train, x_test, y_train, y_test = split(task)
    estimator = pipeline(task, model, prep)
    start = perf_counter()
    estimator.fit(x_train, y_train)
    fit_seconds = perf_counter() - start
    pred = estimator.predict(x_test)
    if task == 'classification':
        metrics = dict(f1_macro=f1_score(y_test, pred, average='macro'),
                       accuracy=accuracy_score(y_test, pred))
    else:
        metrics = dict(mae_ms=mean_absolute_error(y_test, pred),
                       rmse_ms=root_mean_squared_error(y_test, pred))
    return dict(task=task, **metrics, fit_seconds=fit_seconds), pred
