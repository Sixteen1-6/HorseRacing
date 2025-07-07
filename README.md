Horse Racing Position Prediciton:
This repository includes a machine learning system that predicts horse racing finish positons using LightGBM with comprehensive feature engineering and hyperparameter optimzaiton


Overview: This system analyzes historical horse racing data to predict finish positions in upcoming races. It uses advanced feature engineering including:

Jockey-Trainer Combinations: Historical win rates and in-the-money statistics
Track Conditions: Standardized surface and weather conditions
Horse Performance: Past performance metrics and categorization
Race Context: Distance, class level, field size, and post position effects
Temporal Features: Days since last race and seasonal patterns


Our Model Overall Preformance :

RMSE: ~2.16 ± 0.18
MAE: ~1.72 ± 0.15
Top-3 Accuracy: ~68% (predicting horses that finish in top 3)
Win Prediction Accuracy: ~86%


If you want to use this model , create a CSV file with the following :
Required Columns

race_number: Race identifier
horse_name: Horse name
jockey: Jockey name
trainer: Trainer name
weight: Carrying weight
age: Horse age
post_position: Starting gate position

Optional but Recommended

race_type: Type of race (Stakes, Allowance, Claiming, etc.)
purse: Prize money
surface: Track surface (Dirt, Turf, etc.)
distance: Race distance
distance_unit: Distance unit (F=Furlongs, Y=Yards, M=Miles)
track_condition: Track condition (FT=Fast, GD=Good, etc.)
dollar_odds: Betting odds
last_race_date: Date of horse's last race
last_race_finish: Finish position in last race
num_past_starts: Total career starts
num_past_wins: Total career wins
num_past_seconds: Total career second-place finishes
num_past_thirds: Total career third-place finishes



For privacy concerns of our model the repoistory will only include the model PKL files and the .py file that makes the prediciton.

The best parameters that the model using optuna  :
best_params = {
    'objective': 'regression_l1',  # Standard MAE objective
    'metric': 'mae',               # Standard MAE metric
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 800,
    'learning_rate': 0.023395135214485212,
    'num_leaves': 149,
    'max_depth': 13,
    'feature_fraction': 0.6590518544998959,
    'bagging_fraction': 0.8458411332756606,
    'bagging_freq': 2,
    'min_child_samples': 35,
    'lambda_l1': 9.207007688134354,
    'lambda_l2': 5.1272252466576126e-08,
    'random_state': 42,
    'n_jobs': -1
}

```

HorseRacingPredictor/
├── README.md                          # You are here! Complete setup guide
├── predict.py                         # Main prediction script
├── models/                            # Trained model files
│   ├── horse_racing_position_model_cv.pkl
│   ├── horse_racing_preprocessor_cv.pkl
│   └── jockey_trainer_lookup.pkl
├── test_data.csv                      # Your race data goes here
├── requirements.txt                   # Python dependencies
└── .gitignore                         # Git exclusions

```

