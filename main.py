from utils import get_features, cat_cols, float_cols
import argparse 
import pandas as pd
from catboost import CatBoostClassifier
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--path')
    args = parser.parse_args()
    start = time.time()
    df = pd.read_csv(args.path)
    df = get_features(df)
    
    clf_cat =  CatBoostClassifier()
    clf_cat.load_model('model_cat_features.cbm')
    
    clf_float =  CatBoostClassifier()
    clf_float.load_model('model_float_features.cbm')
    
    predict = clf_float.predict_proba(df[float_cols].fillna(0))
    
    result = pd.Series(predict[:, 0], index = df.index)
    result.loc[df[predict < 0.5].index] += 1.1*clf_cat.predict_proba(df[predict < 0.5][cat_cols].fillna(0))[:, 0]
    result.loc[df[predict < 0.5].index] = result.loc[df[predict < 0.5].index]/2
    
    df['answer'] = result < 0.5
    print('done')
    df.to_csv('answer.csv')