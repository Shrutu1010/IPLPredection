import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import warnings
import os
import json
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)

# Set random seed
np.random.seed(42)

# ------------------------
# Data Preprocessing
# ------------------------
def load_and_clean_data():
    try:
        data = pd.read_csv('data/cleaned_IPL_DATASET_2025.csv')
        ball_data = pd.read_csv('data/ball_by_ball_data.csv')
        metadata = pd.read_csv('data/match_metadata.csv')
        
        # Load IPL schedule
        with open('data/ipl_schedule.json', 'r') as f:
            schedule = json.load(f)
        schedule_df = pd.DataFrame(schedule)

        # Load 2024 points table
        points_table = pd.DataFrame([
            {'Team': 'Royal Challengers Bangalore', 'Points': 14, 'Wins': 7, 'NRR': 0.521},
            {'Team': 'Punjab Kings', 'Points': 13, 'Wins': 6, 'NRR': 0.199},
            {'Team': 'Mumbai Indians', 'Points': 12, 'Wins': 6, 'NRR': 0.889},
            {'Team': 'Gujarat Titans', 'Points': 12, 'Wins': 6, 'NRR': 0.748},
            {'Team': 'Delhi Capitals', 'Points': 12, 'Wins': 6, 'NRR': 0.362},
            {'Team': 'Lucknow Super Giants', 'Points': 10, 'Wins': 5, 'NRR': -0.325},
            {'Team': 'Kolkata Knight Riders', 'Points': 9, 'Wins': 4, 'NRR': 0.271},
            {'Team': 'Rajasthan Royals', 'Points': 6, 'Wins': 3, 'NRR': -0.349},
            {'Team': 'Sunrisers Hyderabad', 'Points': 6, 'Wins': 3, 'NRR': -1.103},
            {'Team': 'Chennai Super Kings', 'Points': 4, 'Wins': 2, 'NRR': -1.211}
        ])

        # Rename year column before aggregation
        year_candidates = [col for col in data.columns if any(x in col.lower() for x in ['year', 'season', 'date'])]
        print("Year-related columns:", year_candidates)
        if 'Year' in data.columns:
            data = data.rename(columns={'Year': 'year'})
        elif 'season' in data.columns:
            data = data.rename(columns={'season': 'year'})
        elif 'Season' in data.columns:
            data = data.rename(columns={'Season': 'year'})
        elif 'match_year' in data.columns:
            data = data.rename(columns={'match_year': 'year'})
        elif not year_candidates:
            print("Warning: No year-related column found. Assigning default year 2023.")
            data['year'] = 2023
        else:
            raise ValueError(f"No 'year', 'Year', 'season', or 'Season' column found. Year candidates: {year_candidates}")

        # Aggregate data to team-season level
        agg_columns = {
            'Total Runs Scored by Team': 'sum',
            'Total Balls Faced': 'sum',
            'Batting Average': 'mean',
            'Average Strike Rate': 'mean',
            'Number of Hundreds': 'sum',
            'Number of Fifties': 'sum',
            'Number of Fours': 'sum',
            'Number of Sixes': 'sum',
            'Total Overs Bowled': 'sum',
            'Total Maiden Overs': 'sum',
            'Total Wickets': 'sum',
            'Total Runs Conceded': 'sum',
            'Bowling Average': 'mean',
            'Bowling Economy Rate': 'mean',
            'Average Bowling Strike Rate': 'mean',
            'Total Catches Taken': 'sum',
            'M': 'sum',
            'W': 'sum',
            'L': 'sum',
            'T': 'sum',
            'NR': 'sum',
            'PT': 'sum',
            'NRR': 'mean',
            'Pos': 'min',
            'Result': 'first'
        }
        data = data.groupby(['Team', 'year']).agg(agg_columns).reset_index()
        print("Data shape after aggregation:", data.shape)

        # Enhanced team mapping
        team_mapping = {
            'CSK': 'Chennai Super Kings',
            'MI': 'Mumbai Indians',
            'RCB': 'Royal Challengers Bangalore',
            'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
            'KKR': 'Kolkata Knight Riders',
            'DC': 'Delhi Capitals',
            'Delhi Daredevils': 'Delhi Capitals',
            'DCB': 'Delhi Capitals',
            'SRH': 'Sunrisers Hyderabad',
            'RR': 'Rajasthan Royals',
            'PBKS': 'Punjab Kings',
            'PK': 'Punjab Kings',
            'Kings XI Punjab': 'Punjab Kings',
            'Punjab Kings XI': 'Punjab Kings',
            'Punjab': 'Punjab Kings',
            'PW': 'Punjab Kings',
            'LSG': 'Lucknow Super Giants',
            'GT': 'Gujarat Titans',
            'KRR': 'Kolkata Knight Riders',
            'GL': 'Gujarat Titans',
            'RPG': 'Rajasthan Royals',
            'RPSG': 'Rajasthan Royals'
        }
        print("Original teams in data:", data['Team'].unique())
        data['Team'] = data['Team'].map(team_mapping).fillna(data['Team'])
        print("Mapped teams in data:", data['Team'].unique())

        # Filter active teams
        active_teams = [
            'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
            'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
        ]
        data = data[data['Team'].isin(active_teams)]
        print("Teams after filtering:", data['Team'].unique())

        # Standardize other datasets
        ball_data['batting_team'] = ball_data['batting_team'].map(team_mapping).fillna(ball_data['batting_team'])
        ball_data['bowling_team'] = ball_data['bowling_team'].map(team_mapping).fillna(ball_data['bowling_team'])
        metadata['team1'] = metadata['team1'].map(team_mapping).fillna(metadata['team1'])
        metadata['team2'] = metadata['team2'].map(team_mapping).fillna(metadata['team2'])
        metadata['winner'] = metadata['winner'].map(team_mapping).fillna(metadata['winner'])
        schedule_df['team1'] = schedule_df['match'].str.split(' vs ').str[0].map(team_mapping).fillna(schedule_df['match'].str.split(' vs ').str[0])
        schedule_df['team2'] = schedule_df['match'].str.split(' vs ').str[1].map(team_mapping).fillna(schedule_df['match'].str.split(' vs ').str[1])
        points_table['Team'] = points_table['Team'].map(team_mapping).fillna(points_table['Team'])

        # Handle missing values
        data = data.fillna({
            'year': 2023,
            'NRR': 0,
            'Total Runs Scored by Team': 0,
            'Total Balls Faced': 0,
            'Batting Average': 0,
            'Average Strike Rate': 0,
            'Number of Hundreds': 0,
            'Number of Fifties': 0,
            'Number of Fours': 0,
            'Number of Sixes': 0,
            'Total Overs Bowled': 0,
            'Total Maiden Overs': 0,
            'Total Wickets': 0,
            'Total Runs Conceded': 0,
            'Bowling Average': 0,
            'Bowling Economy Rate': 0,
            'Average Bowling Strike Rate': 0,
            'Total Catches Taken': 0,
            'Result': 'Unknown',
            'Pos': 10,
            'M': 0,
            'W': 0,
            'L': 0,
            'T': 0,
            'NR': 0,
            'PT': 0,
            'NRR_points': 0,
            'Form': 'Unknown'
        })

        ball_data = ball_data.fillna({
            'extra_runs': 0,
            'batsman_runs': 0,
            'total_runs': 0,
            'is_wicket': 0
        })

        metadata = metadata.fillna({
            'city': 'Unknown',
            'player_of_match': 'Unknown',
            'result': 'Unknown',
            'result_margin': 0,
            'target_runs': 0,
            'target_overs': 0
        })

        # Extract year from date in metadata
        if 'date' in metadata.columns:
            metadata['year'] = pd.to_datetime(metadata['date'], errors='coerce').dt.year
        elif 'season' in metadata.columns:
            metadata['year'] = metadata['season'].astype(int)
        else:
            print("Warning: No 'year' or 'date' column in metadata. Using all data for venue_win_rate.")
            metadata['year'] = 2023

        return data, ball_data, metadata, schedule_df, points_table

    except FileNotFoundError as e:
        print(f"Error: {e}. Ensure all files are in the 'data/' directory.")
        raise
    except Exception as e:
        print(f"Error in load_and_clean_data: {e}")
        raise

# ------------------------
# Feature Engineering
# ------------------------
def create_features(data, ball_data, metadata, schedule_df, points_table):
    try:
        # Verify no duplicates after aggregation
        print("Shape after aggregation:", data.shape)
        print("Duplicate team-year pairs:", data.duplicated(subset=['Team', 'year']).sum())
        data = data.drop_duplicates(subset=['Team', 'year'])
        print("Shape after dropping duplicates (post-aggregation):", data.shape)

        # Team-level features
        data['batting_strength'] = data['Average Strike Rate'] / (data['Batting Average'].replace(0, 1))
        data['bowling_strength'] = data['Bowling Economy Rate'].replace(0, 1) / data['Average Bowling Strike Rate'].replace(0, 1)
        data['consistency'] = data['Number of Fifties'] / (data['M'].replace(0, 1))

        # Historical win rate (pre-2024)
        data['historical_win_rate'] = data['W'] / data['M'].replace(0, 1)
        data['historical_win_rate'] = data['historical_win_rate'].fillna(data['historical_win_rate'].mean())

        # Merge 2024 points table
        print("Teams in data before merge:", data['Team'].unique())
        print("Teams in points_table:", points_table['Team'].unique())
        points_table = points_table.drop_duplicates(subset=['Team'])
        data = data.merge(points_table[['Team', 'Wins', 'NRR']], on='Team', how='left')
        print("Columns after merge:", data.columns.tolist())
        print("Sample data after merge:", data[['Team', 'year', 'Wins', 'NRR_y']].head(10))
        data = data.drop_duplicates(subset=['Team', 'year'])
        print("Shape after dropping duplicates (post-merge):", data.shape)

        data['recent_form'] = data['Wins'] / 10 + data['NRR_y']
        data['recent_form'] = data['recent_form'].fillna(data['historical_win_rate'])

        # Head-to-head win rate (pre-2024)
        if 'year' in metadata.columns:
            historical_matches = metadata[metadata['year'] < 2024]
            head_to_head = historical_matches.groupby(['team1', 'team2', 'winner'])['id'].count().unstack().fillna(0)
            head_to_head['win_rate'] = head_to_head[head_to_head.columns[0]] / (head_to_head[head_to_head.columns[0]] + head_to_head[head_to_head.columns[1]])
            head_to_head = head_to_head['win_rate'].reset_index()
            data = data.merge(head_to_head, left_on='Team', right_on='team1', how='left')
            data['head_to_head_win_rate'] = data['win_rate'].fillna(0.5)
            data = data.drop(columns=['team1', 'team2', 'win_rate'])
        else:
            data['head_to_head_win_rate'] = 0.5

        # Venue performance
        if 'year' in metadata.columns:
            venue_wins = metadata[metadata['year'] < 2024].groupby(['venue', 'winner'])['id'].count().reset_index()
            venue_wins['venue_win_rate'] = venue_wins.groupby('venue')['id'].transform(lambda x: x / x.sum())
            venue_wins = venue_wins.rename(columns={'winner': 'Team'})
            data = data.merge(venue_wins[['venue', 'Team', 'venue_win_rate']], on='Team', how='left')
            data['venue_win_rate'] = data['venue_win_rate'].fillna(0.5)
        else:
            data['venue_win_rate'] = 0.5

        return data

    except Exception as e:
        print(f"Error in create_features: {e}")
        raise

# ------------------------
# Simulate 2025 Season
# ------------------------
def simulate_match(team1, team2, venue, model, scaler, feature_names, historical_data, points_table):
    try:
        team1_data = historical_data[historical_data['Team'] == team1].tail(1)[feature_names]
        team2_data = historical_data[historical_data['Team'] == team2].tail(1)[feature_names]
        
        if team1_data.empty or team2_data.empty:
            print(f"Warning: No data for {team1} or {team2}. Using recent_form.")
            team1_form = historical_data[historical_data['Team'] == team1]['recent_form'].iloc[-1] if team1 in historical_data['Team'].values else 0.5
            team2_form = historical_data[historical_data['Team'] == team2]['recent_form'].iloc[-1] if team2 in historical_data['Team'].values else 0.5
            return team1 if team1_form > team2_form else team2

        team1_data = scaler.transform(team1_data)
        team2_data = scaler.transform(team2_data)

        prob1 = model.predict_proba(team1_data)[0][0]
        prob2 = model.predict_proba(team2_data)[0][0]

        # Adjust probability by 2024 performance
        team1_form = historical_data[historical_data['Team'] == team1]['recent_form'].iloc[-1] if team1 in historical_data['Team'].values else 0.5
        team2_form = historical_data[historical_data['Team'] == team2]['recent_form'].iloc[-1] if team2 in historical_data['Team'].values else 0.5
        prob1 = prob1 * (1 + 3 * team1_form)
        prob2 = prob2 * (1 + 3 * team2_form)
        
        return team1 if prob1 > prob2 else team2

    except Exception as e:
        print(f"Error in simulate_match: {e}")
        return team1 if np.random.rand() > 0.5 else team2

def simulate_season(schedule_df, model, scaler, feature_names, historical_data, points_table):
    try:
        active_teams = [
            'Chennai Super Kings', 'Mumbai Indians', 'Royal Challengers Bangalore',
            'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
            'Rajasthan Royals', 'Punjab Kings', 'Lucknow Super Giants', 'Gujarat Titans'
        ]
        points_table_sim = {team: {'points': 0, 'matches': 0} for team in active_teams}
        print("Number of matches in schedule:", len(schedule_df))
        
        for _, match in schedule_df.iterrows():
            team1 = match['team1']
            team2 = match['team2']
            venue = match['venue']
            
            if team1 not in points_table_sim or team2 not in points_table_sim:
                print(f"Warning: {team1} or {team2} not in points_table_sim. Skipping match.")
                continue
                
            winner = simulate_match(team1, team2, venue, model, scaler, feature_names, historical_data, points_table)
            points_table_sim[winner]['points'] += 2
            points_table_sim[team1]['matches'] += 1
            points_table_sim[team2]['matches'] += 1
        
        rankings = pd.DataFrame(points_table_sim).T.reset_index().rename(columns={'index': 'Team'})
        rankings = rankings.sort_values('points', ascending=False).reset_index(drop=True)
        
        rankings['Result'] = 'Other'
        rankings.loc[0, 'Result'] = 'Winner'
        rankings.loc[1, 'Result'] = 'Runner-up'
        rankings.loc[2, 'Result'] = 'Second Runner-up'
        rankings.loc[3, 'Result'] = 'Eliminator'
        
        return rankings

    except Exception as e:
        print(f"Error in simulate_season: {e}")
        raise

# ------------------------
# Model Training and Evaluation
# ------------------------
def train_and_evaluate_models(X, y, feature_names):
    try:
        print("Data shape:", X.shape)
        print("Result counts:", y.value_counts())
        
        if 'year' in X.columns:
            train_mask = X['year'] < 2024
            test_mask = X['year'] == 2024
            X_train = X[train_mask][feature_names]
            y_train = y[train_mask]
            X_test = X[test_mask][feature_names]
            y_test = y[test_mask]
            print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X[feature_names], y, test_size=0.2, random_state=42, stratify=y)
            print("Warning: No 'year' column. Using random train-test split.")
            print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

        # Undersample majority class ("Other")
        undersampler = RandomUnderSampler(sampling_strategy={'Other': 5000}, random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        print("Class distribution after undersampling:", Counter(y_train))

        # Apply SMOTE to minority classes
        smote = SMOTE(sampling_strategy={'Eliminator': 5000, 'Winner': 9600, 'Second Runner-up': 8640}, k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:", Counter(y_train))

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test) if len(y_test) > 0 else y_train[:1]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test) if X_test.shape[0] > 0 else X_train[:1]

        n_classes = len(np.unique(y_train))
        models = {
            'Random Forest': RandomForestClassifier(
                random_state=42,
                max_depth=3,
                min_samples_split=15,
                min_samples_leaf=7,
                n_estimators=100
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=42,
                max_depth=2,
                learning_rate=0.005,
                n_estimators=30,
                min_samples_split=20
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                max_depth=2,
                learning_rate=0.005,
                n_estimators=30,
                min_child_weight=5
            )
        }

        results = {}
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test) if X_test.shape[0] > 0 else model.predict(X_train[:1])
            results[name] = {
                'CV Accuracy': cv_scores.mean(),
                'CV Accuracy Std': cv_scores.std(),
                'Test Accuracy': accuracy_score(y_test, y_pred) if len(y_test) > 0 else np.nan,
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0) if len(y_test) > 0 else np.nan,
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0) if len(y_test) > 0 else np.nan,
                'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0) if len(y_test) > 0 else np.nan
            }

        nn_model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(8, activation='relu'),
            Dropout(0.5),
            Dense(n_classes, activation='softmax')
        ])
        nn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        nn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        y_pred = np.argmax(nn_model.predict(X_test), axis=1) if X_test.shape[0] > 0 else np.argmax(nn_model.predict(X_train[:1]), axis=1)
        results['Neural Network'] = {
            'CV Accuracy': np.nan,
            'CV Accuracy Std': np.nan,
            'Test Accuracy': accuracy_score(y_test, y_pred) if len(y_test) > 0 else np.nan,
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0) if len(y_test) > 0 else np.nan,
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0) if len(y_test) > 0 else np.nan,
            'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0) if len(y_test) > 0 else np.nan
        }

        best_model_name = max(results, key=lambda x: results[x]['CV Accuracy'] if not np.isnan(results[x]['CV Accuracy']) else results[x]['Test Accuracy'])
        best_model = models.get(best_model_name, nn_model)
        best_model.fit(X_train, y_train)

        joblib.dump(best_model, 'model/best_model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(le, 'model/label_encoder.pkl')

        metrics_df = pd.DataFrame(results).T
        metrics_df[['CV Accuracy', 'Test Accuracy']].plot(kind='bar', figsize=(10, 6))
        plt.title('Model Performance Comparison')
        plt.savefig('static/model_performance.png')
        plt.close()

        if best_model_name in ['Random Forest', 'XGBoost']:
            importances = best_model.feature_importances_
            plt.bar(feature_names, importances)
            plt.title('Feature Importance')
            plt.xticks(rotation=45)
            plt.savefig('static/feature_importance.png')
            plt.close()

        return results, best_model_name

    except Exception as e:
        print(f"Error in train_and_evaluate_models: {e}")
        raise

# ------------------------
# Main Execution
# ------------------------
if __name__ == "__main__":
    try:
        data, ball_data, metadata, schedule_df, points_table = load_and_clean_data()
        data = create_features(data, ball_data, metadata, schedule_df, points_table)

        features = [
            'batting_strength',
            'bowling_strength',
            'consistency',
            'historical_win_rate',
            'head_to_head_win_rate',
            'venue_win_rate'
        ]
        X = data[(['year'] if 'year' in data.columns else []) + features]
        data['Result'] = data['Pos'].map({
            1: 'Winner',
            2: 'Runner-up',
            3: 'Second Runner-up',
            4: 'Eliminator'
        }).fillna('Other')
        y = data['Result']

        results, best_model_name = train_and_evaluate_models(X, y, features)
        print(f"Best Model: {best_model_name}")
        print("Model Performance:")
        for model, metrics in results.items():
            print(f"{model}: {metrics}")

        model = joblib.load('model/best_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        rankings = simulate_season(schedule_df, model, scaler, features, data, points_table)
        print("\nPredicted IPL 2025 Rankings:")
        print(rankings[['Team', 'Result', 'points']])

        rankings.to_csv('data/predicted_ipl_2025_rankings.csv', index=False)

        if 'year' in data.columns:
            sns.lineplot(x='year', y='batting_strength', hue='Team', data=data)
            plt.savefig('static/batting_strength_trend.png')
            plt.close()

    except Exception as e:
        print(f"Main execution failed: {e}")
        raise