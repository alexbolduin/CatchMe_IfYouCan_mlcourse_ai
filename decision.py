# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import os
import json
from tqdm import tqdm_notebook
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import collections

PATH_TO_DATA = '../mlcourse.ai - Spring 2019/data/dota2/'

target = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                            'train_targets.csv'), 
                                   index_col='match_id_hash')

y = target['radiant_win'].map({True:1, False:0})

def read_matches(matches_file):
    
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)
    
    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)
            
MATCH_FEATURES = [
    ('game_time', lambda m: m['game_time']),
    ('game_mode', lambda m: m['game_mode']),
    ('lobby_type', lambda m: m['lobby_type']),
    ('objectives_len', lambda m: len(m['objectives'])),
    ('chat_len', lambda m: len(m['chat'])),
]

PLAYER_FIELDS = [
    'hero_id',
    'hero_name',
    'account_id_hash',
    
    'kills',
    'deaths',
    'assists',
    'denies',
    
    'gold',
    'lh',
    'xp',
    'health',
    'max_health',
    'max_mana',
    'level',

    'x',
    'y',
    
    'stuns',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'teamfight_participation',
    'towers_killed',
    'roshans_killed',
    'obs_placed',
    'sen_placed',
]

def extract_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))
        row.append((f'{player_name}_ability_level', 
                    len(player['ability_upgrades'])))
        row.append((f'{player_name}_max_hero_hit', 
                    player['max_hero_hit']['value']))
        row.append((f'{player_name}_purchase_count', 
                    len(player['purchase_log'])))
        row.append((f'{player_name}_count_ability_use', 
                    sum(player['ability_uses'].values())))
        row.append((f'{player_name}_damage_dealt', 
                    sum(player['damage'].values())))
        row.append((f'{player_name}_damage_received', 
                    sum(player['damage_taken'].values())))
        row.append((f'{player_name}_buyback_log', 
                    len(player['buyback_log'])))
        row.append((f'{player_name}_gold_reasons', 
                    len(player['gold_reasons'])))
        row.append((f'{player_name}_xp_reasons', 
                    len(player['xp_reasons'])))
            
    return collections.OrderedDict(row)

df_new_features = []
for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    features = extract_features_csv(match)
    
    df_new_features.append(features)
    
df_new_features = pd.DataFrame.from_records(
        df_new_features).set_index('match_id_hash')

test_new_features = []
for match in read_matches(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    features = extract_features_csv(match)
    
    test_new_features.append(features)
    
test_new_features = pd.DataFrame.from_records(
        test_new_features).set_index('match_id_hash')

r_train_hero_name = df_new_features[['r1_hero_name', 'r2_hero_name', 
                                     'r3_hero_name', 'r4_hero_name', 
                                     'r5_hero_name']]
r_train_account = df_new_features[['r1_account_id_hash', 'r2_account_id_hash', 
                                   'r3_account_id_hash', 'r4_account_id_hash', 
                                   'r5_account_id_hash']]

d_train_hero_name = df_new_features[['d1_hero_name', 'd2_hero_name', 
                                     'd3_hero_name', 'd4_hero_name', 
                                     'd5_hero_name']]
d_train_account = df_new_features[['d1_account_id_hash', 'd2_account_id_hash', 
                                   'd3_account_id_hash', 'd4_account_id_hash', 
                                   'd5_account_id_hash']]

r_test_hero_name = test_new_features[['r1_hero_name', 'r2_hero_name', 
                                      'r3_hero_name', 'r4_hero_name', 
                                      'r5_hero_name']]
r_test_account = test_new_features[['r1_account_id_hash', 'r2_account_id_hash', 
                                    'r3_account_id_hash', 'r4_account_id_hash', 
                                    'r5_account_id_hash']]

d_test_hero_name = test_new_features[['d1_hero_name', 'd2_hero_name', 
                                      'd3_hero_name', 'd4_hero_name', 
                                      'd5_hero_name']]
d_test_account = test_new_features[['d1_account_id_hash', 'd2_account_id_hash', 
                                    'd3_account_id_hash', 'd4_account_id_hash', 
                                    'd5_account_id_hash']]

r_cols = ['r1_hero_name', 'r2_hero_name', 'r3_hero_name', 'r4_hero_name', 
          'r5_hero_name']
r_id_cols = ['r1_account_id_hash', 'r2_account_id_hash', 'r3_account_id_hash', 
             'r4_account_id_hash', 'r5_account_id_hash']

d_cols = ['d1_hero_name', 'd2_hero_name', 'd3_hero_name', 'd4_hero_name', 
          'd5_hero_name']
d_id_cols = ['d1_account_id_hash', 'd2_account_id_hash', 'd3_account_id_hash', 
             'd4_account_id_hash', 'd5_account_id_hash']

r_ce_poly = ce.PolynomialEncoder(cols=r_cols)
d_ce_poly = ce.PolynomialEncoder(cols=d_cols)

r_id_ce_hash = ce.OrdinalEncoder(cols=r_id_cols)
d_id_ce_hash = ce.OrdinalEncoder(cols=d_id_cols)

r_ce_poly.fit(r_train_hero_name[r_cols], y)
d_ce_poly.fit(d_train_hero_name[d_cols], y)
r_id_ce_hash.fit(r_train_account[r_id_cols], y)
d_id_ce_hash.fit(d_train_account[d_id_cols], y)

df_new_features.drop(['r1_hero_name', 'r2_hero_name', 'r3_hero_name', 
                      'r4_hero_name', 'r5_hero_name', 
                      'd1_hero_name', 'd2_hero_name', 'd3_hero_name', 
                      'd4_hero_name', 'd5_hero_name'], 
                      axis=1, inplace=True)
test_new_features.drop(['r1_hero_name', 'r2_hero_name', 'r3_hero_name', 
                        'r4_hero_name', 'r5_hero_name', 
                      'd1_hero_name', 'd2_hero_name', 'd3_hero_name', 
                      'd4_hero_name', 'd5_hero_name'], 
                      axis=1, inplace=True)

df_new_features.drop(['r1_account_id_hash', 'r2_account_id_hash', 
                      'r3_account_id_hash', 'r4_account_id_hash', 
                      'r5_account_id_hash', 
                      'd1_account_id_hash', 'd2_account_id_hash', 
                      'd3_account_id_hash', 'd4_account_id_hash', 
                      'd5_account_id_hash'], 
                      axis=1, inplace=True)
test_new_features.drop(['r1_account_id_hash', 'r2_account_id_hash', 
                        'r3_account_id_hash', 'r4_account_id_hash', 
                        'r5_account_id_hash', 
                        'd1_account_id_hash', 'd2_account_id_hash', 
                        'd3_account_id_hash', 'd4_account_id_hash', 
                        'd5_account_id_hash'], 
                      axis=1, inplace=True)

for c in ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 
          'max_health', 'max_mana', 'level', 'x', 'y', 'stuns', 
          'creeps_stacked', 'camps_stacked', 'rune_pickups', 
          'firstblood_claimed', 'teamfight_participation', 'towers_killed', 
          'roshans_killed', 'obs_placed', 'sen_placed', 'ability_level', 
          'max_hero_hit', 'purchase_count', 'count_ability_use', 
          'damage_dealt', 'damage_received', 'buyback_log', 'gold_reasons', 
          'xp_reasons']:
    
    r_columns = [f'r{i}_{c}' for i in range(1, 6)]
    d_columns = [f'd{i}_{c}' for i in range(1, 6)]
    
    df_new_features['r_total_' + c] = df_new_features[r_columns].sum(1)
    df_new_features['d_total_' + c] = df_new_features[d_columns].sum(1)
    df_new_features['total_' + c + '_ratio'] = (
            df_new_features['r_total_' + c] + 1) / (
                    df_new_features['d_total_' + c] + 1)
    
    test_new_features['r_total_' + c] = test_new_features[r_columns].sum(1)
    test_new_features['d_total_' + c] = test_new_features[d_columns].sum(1)
    test_new_features['total_' + c + '_ratio'] = (
            test_new_features['r_total_' + c] + 1) / (
                    test_new_features['d_total_' + c] + 1)
    
    df_new_features['r_std_' + c] = df_new_features[r_columns].std(1)
    df_new_features['d_std_' + c] = df_new_features[d_columns].std(1)
    df_new_features['std_' + c + '_ratio'] = (
            df_new_features['r_std_' + c] + 1) / (
                    df_new_features['d_std_' + c] + 1)
    
    test_new_features['r_std_' + c] = test_new_features[r_columns].std(1)
    test_new_features['d_std_' + c] = test_new_features[d_columns].std(1)
    test_new_features['std_' + c + '_ratio'] = (
            test_new_features['r_std_' + c] + 1) / (
                    test_new_features['d_std_' + c] + 1)
    
    df_new_features['r_mean_' + c] = df_new_features[r_columns].mean(1)
    df_new_features['d_mean_' + c] = df_new_features[d_columns].mean(1)
    df_new_features['mean_' + c + '_ratio'] = (
            df_new_features['r_mean_' + c] + 1) / (
                    df_new_features['d_mean_' + c] + 1)
    
    test_new_features['r_mean_' + c] = test_new_features[r_columns].mean(1)
    test_new_features['d_mean_' + c] = test_new_features[d_columns].mean(1)
    test_new_features['mean_' + c + '_ratio'] = (
            test_new_features['r_mean_' + c] + 1) / (
                    test_new_features['d_mean_' + c] + 1)
    
scaler = StandardScaler().fit(df_new_features.fillna(0), y)

df_scal_features = pd.DataFrame(
        scaler.fit_transform(
                df_new_features.fillna(0), 
                y), 
                index=df_new_features.index, 
                columns=df_new_features.columns)
test_scal_features = pd.DataFrame(
        scaler.transform(
                test_new_features.fillna(0)), 
                index=test_new_features.index, 
                columns=test_new_features.columns)

df_train_concat = pd.concat([df_scal_features.fillna(0), 
                             r_ce_poly.fit_transform
                             (r_train_hero_name[r_cols], y),
                             d_ce_poly.fit_transform
                             (d_train_hero_name[d_cols], y),
                             r_id_ce_hash.fit_transform
                             (r_train_account[r_id_cols], y),
                             d_id_ce_hash.fit_transform
                             (d_train_account[d_id_cols], y),
                            ], axis=1).fillna(0)
    
df_test_concat = pd.concat([test_scal_features.fillna(0), 
                            r_ce_poly.transform
                            (r_test_hero_name[r_cols]),
                            d_ce_poly.transform
                            (d_test_hero_name[d_cols]),
                            r_id_ce_hash.transform
                            (r_test_account[r_id_cols]),
                            d_id_ce_hash.transform
                            (d_test_account[d_id_cols]),
                           ], axis=1).fillna(0)

X_new = df_train_concat.reset_index(drop=True)
X_test_new = df_test_concat.copy().reset_index(drop=True)

lr = 0.09552 / 3.3

cat = CatBoostClassifier(random_state=17, 
                         iterations=2321,
                         learning_rate=lr,
                         verbose=100,
                         custom_metric='AUC',
                         eval_metric='AUC',
                         use_best_model=False,
                         early_stopping_rounds=200)

cat.fit(X_new,
        y,
        verbose=False,
        plot=False)

y_test_pred = cat.predict_proba(X_test_new)[:, 1]

df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, 
                                 index=df_test_concat.index)

submission_filename = 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_cat'))

df_submission.to_csv(submission_filename)
print('Submission saved to {}'.format(submission_filename))
