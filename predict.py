import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Helper functions (copied from the original script to ensure consistency)
def parse_time_to_seconds(time_str):
    """Convert a time string (MM:SS.s or S.s) to seconds."""
    if pd.isna(time_str):
        return np.nan
    if isinstance(time_str, (int, float)):
        return float(time_str)

    time_str = str(time_str).strip()
    match_min_sec = re.match(r'(\d{1,2}):(\d{1,2}(?:\.\d+)?)', time_str)
    match_sec = re.match(r'(\d+(?:\.\d+)?)', time_str)

    if match_min_sec:
        minutes = float(match_min_sec.group(1))
        seconds = float(match_min_sec.group(2))
        return minutes * 60 + seconds
    elif match_sec:
        return float(match_sec.group(1))
    else:
        return np.nan

def calculate_days_between(date1, date2):
    """Calculate days between two dates (robust to NaT)."""
    if pd.isna(date1) or pd.isna(date2):
        return np.nan
    d1 = pd.to_datetime(date1, errors='coerce')
    d2 = pd.to_datetime(date2, errors='coerce')
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    return (d1 - d2).days

def categorize_horses(df):
    """Categorize horses based on various attributes."""
    # Create horse categories based on age
    if 'age' in df.columns:
        # Ensure age is numeric before cutting
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age_category'] = pd.cut(
            df['age'],
            bins=[0, 3, 5, 7, 100],
            labels=['Young', 'Prime', 'Mature', 'Veteran'],
            right=False # Include 0 in the first bin
        )

    # Create categories based on past performance
    if all(col in df.columns for col in ['num_past_wins', 'num_past_starts']):
         # Ensure inputs are numeric
        df['num_past_wins'] = pd.to_numeric(df['num_past_wins'], errors='coerce').fillna(0)
        df['num_past_starts'] = pd.to_numeric(df['num_past_starts'], errors='coerce').fillna(0)
        # Calculate win rate
        df['win_rate'] = df['num_past_wins'] / df['num_past_starts'].clip(lower=1) # Avoid division by zero
        # Categorize by win rate
        df['win_rate_category'] = pd.cut(
            df['win_rate'],
            bins=[-0.001, 0.1, 0.2, 0.3, 1.01], # Adjust bins slightly for edge cases
            labels=['Low', 'Medium', 'High', 'Elite'],
            right=False
        )

    # Create categories based on odds (if available)
    if 'dollar_odds' in df.columns:
        df['dollar_odds'] = pd.to_numeric(df['dollar_odds'], errors='coerce')
        df['odds_category'] = pd.cut(
            df['dollar_odds'],
            bins=[-0.001, 2, 5, 10, 1000],
            labels=['Favorite', 'Contender', 'Longshot', 'Huge Longshot'],
            right=False
        )

    # Create experience categories
    if 'num_past_starts' in df.columns:
        df['experience_category'] = pd.cut(
            df['num_past_starts'],
            bins=[-1, 5, 15, 30, 1000],
            labels=['Novice', 'Experienced', 'Veteran', 'Elite'],
            right=False
        )

    return df

def main():
    # File paths
    model_path = 'models/horse_racing_position_model_cv.pkl'
    preprocessor_path = 'models/horse_racing_preprocessor_cv.pkl'
    jt_lookup_path = 'models/jockey_trainer_lookup.pkl'
    new_data_path = 'test_data.csv'

    # Track default values used
    default_values_used = {}
    columns_with_defaults = []

    # Check if model and preprocessor files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please make sure to run the training script first.")
        return

    if not os.path.exists(preprocessor_path):
        print(f"Error: Preprocessor file '{preprocessor_path}' not found. Please make sure to run the training script first.")
        return

    if not os.path.exists(new_data_path):
        print(f"Error: Data file '{new_data_path}' not found. Please check the file path.")
        return

    # Check for jockey-trainer lookup table
    use_jt_lookup = os.path.exists(jt_lookup_path)
    jt_lookup = None
    if use_jt_lookup:
        print(f"Found jockey-trainer lookup table. Will use historical statistics.")
        try:
            jt_lookup = joblib.load(jt_lookup_path)
            print(f"Loaded statistics for {len(jt_lookup['win_rates'])} jockey-trainer combinations")
        except Exception as e:
            print(f"Error loading jockey-trainer lookup table: {e}")
            use_jt_lookup = False
    else:
        print("Jockey-trainer lookup table not found. Will use default values based on racing industry averages.")

    # Load model and preprocessor
    print("Loading model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Load new data
    print(f"Loading new race data from {new_data_path}...")
    try:
        df = pd.read_csv(new_data_path, low_memory=False)
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        return

    # Check if we have race data
    if df.empty:
        print("Error: The data file is empty.")
        return

    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Store original horse names for final output
    df['original_horse_name'] = df['horse_name'].copy()

    # Create a unique race identifier
    if 'race_number' in df.columns:
        print("Creating race identifiers...")
        # Create a new sequential race ID based on race_number
        df = df.sort_values('race_number')
        df['race_id_temp'] = (df['race_number'] != df['race_number'].shift(1)).cumsum()
        df['race_id'] = 'race_' + df['race_id_temp'].astype(str)
        df = df.drop(columns=['race_id_temp'])
    else:
        print("Warning: No race_number column. Creating artificial race groups.")
        # Create artificial race groups of approximately 8 horses
        df['race_id'] = 'race_' + (df.index // 8).astype(str)

    # Store race details for output
    race_details = df.groupby('race_id').first().reset_index()
    race_details = race_details[['race_id', 'race_number']]

    # Apply the same preprocessing steps as in training
    print("Applying feature engineering...")

    # Basic Preprocessing
    print("Step 1: Basic Preprocessing (Dates, Basic Missing Values, Data Types)")
    date_columns = ['last_race_date', 'horse_dob']
    for col in date_columns:
        if col in df.columns:
            print(f"Converting '{col}' to datetime...")
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # We need to keep race_date for some feature generation but drop it later
    if 'race_date' in df.columns:
        print("Converting 'race_date' to datetime (temporary)...")
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')

    time_cols = ['win_time', 'finish_time', 'last_race_time', 'second_call_time', 'final_time']
    for col in time_cols:
        if col in df.columns:
            new_col_name = f"{col}_seconds"
            print(f"Converting '{col}' to seconds (column '{new_col_name}')...")
            df[new_col_name] = df[col].apply(parse_time_to_seconds)
            # Drop original time column after conversion
            df = df.drop(columns=[col], errors='ignore')

    categorical_cols = ['surface', 'track_condition', 'weather', 'breed',
                       'sex', 'medication', 'track_name',
                       'jockey', 'trainer', 'owner', 'horse_name', 'race_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Handle age
    if 'age' in df.columns and df['age'].notna().any():
        print("Using existing age values from dataset")
        # Ensure age is numeric
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        print("Warning: 'horse_dob' or 'race_date' missing. Cannot calculate 'age'.")
        df['age'] = np.nan  # Create age column with NaNs if source columns are missing
        default_values_used['age'] = {'default_value': 5, 'reason': 'Missing source columns'}
        columns_with_defaults.append('age')

    print("Step 2: Feature Engineering")

    # Preserve the days_since_last_race calculation since it's a valuable feature
    if 'race_date' in df.columns and 'last_race_date' in df.columns:
        print("Calculating 'days_since_last_race'...")
        df['days_since_last_race'] = df.apply(
            lambda row: calculate_days_between(row['race_date'], row['last_race_date']),
            axis=1
        )
        # Fill NaNs after calculation
        median_days = df['days_since_last_race'].median()
        if pd.isna(median_days):  # If all values are NaN, use a default
            median_days = 30  # Default value if no data is available
            default_values_used['days_since_last_race'] = {'default_value': 30, 'reason': 'All values were NaN'}
            columns_with_defaults.append('days_since_last_race')
        df['days_since_last_race'] = df['days_since_last_race'].fillna(median_days)
        print(f"Imputed missing 'days_since_last_race' with median: {median_days:.2f}")
    else:
        print("Warning: Cannot calculate 'days_since_last_race' due to missing date columns.")
        df['days_since_last_race'] = 30  # Default value
        default_values_used['days_since_last_race'] = {'default_value': 30, 'reason': 'Missing source columns'}
        columns_with_defaults.append('days_since_last_race')

    if 'track_condition' in df.columns:
        print("Standardizing 'track_condition'...")
        conditions_map = {
            'FT': 'Fast', 'GD': 'Good', 'SY': 'Sloppy', 'MY': 'Muddy',
            'HY': 'Heavy', 'SL': 'Slow', 'FM': 'Firm', 'YL': 'Yielding',
            'SF': 'Soft', 'HD': 'Hard', 'WF': 'Wet Fast', 'FZ': 'Frozen'  # Added more common codes
        }
        # Ensure track_condition is string before mapping
        df['standard_condition'] = df['track_condition'].astype(str).str.upper().map(conditions_map).fillna(df['track_condition'].astype(str))
        df['standard_condition'] = df['standard_condition'].astype('category')

    if 'race_type' in df.columns and 'purse' in df.columns:
        print("Identifying 'class_level' based on race type and purse...")
        df['purse'] = pd.to_numeric(df['purse'], errors='coerce').fillna(0)  # Ensure purse is numeric, fill NaNs with 0

        def determine_class(row):
            race_type = str(row.get('race_type', '')).upper()
            purse = row.get('purse', 0)

            # Prioritize race type keywords
            if any(k in race_type for k in ['STAKES', 'STK', 'GRADED', 'GRD', 'G1', 'G2', 'G3', 'LISTED', 'LST']): return 1
            if 'ALLOWANCE' in race_type or 'ALW' in race_type: return 2
            if 'MAIDEN SPECIAL WEIGHT' in race_type or 'MSW' in race_type: return 3
            if 'MAIDEN CLAIMING' in race_type or 'MCL' in race_type: return 4
            if 'CLAIMING' in race_type or 'CLM' in race_type:
                 # Claiming levels based on purse
                if purse >= 20000: return 4
                if purse >= 10000: return 5
                return 6  # Lower level claiming
            if 'STARTER' in race_type: return 4  # Often similar to mid-level claiming/allowance
            # Fallback based on purse only if type is unclear
            if purse >= 100000: return 1
            if purse >= 50000: return 2
            if purse >= 25000: return 3
            if purse >= 15000: return 4
            if purse >= 10000: return 5
            return 6  # Lowest level / Unknown

        df['class_level'] = df.apply(determine_class, axis=1).astype('category')

    print("Calculating 'num_horses_in_race'...")
    if 'race_number' in df.columns:
        print("Using race_number to estimate horses in race...")
        # Use race_number alone
        df['num_horses_in_race'] = df.groupby('race_number', observed=False)['horse_name'].transform('size')
        df['num_horses_in_race'] = df['num_horses_in_race'].fillna(df['num_horses_in_race'].median())
    else:
        print("Warning: Cannot calculate 'num_horses_in_race' due to missing race_number.")
        df['num_horses_in_race'] = df.get('num_runners', np.nan)  # Fallback if num_runners exists
        if df['num_horses_in_race'].isnull().all():
            print("No 'num_runners' column available. Using default value of 8 horses per race.")
            df['num_horses_in_race'] = 8  # Default imputation if all else fails
            default_values_used['num_horses_in_race'] = {'default_value': 8, 'reason': 'Missing source columns'}
            columns_with_defaults.append('num_horses_in_race')

    if 'post_position' in df.columns and 'num_horses_in_race' in df.columns:
        print("Calculating post position features...")
        df['post_position'] = pd.to_numeric(df['post_position'], errors='coerce')
        # Avoid division by zero or NaN if num_horses_in_race is 1 or less, or missing
        valid_norm = (df['num_horses_in_race'] > 1) & df['post_position'].notna()
        df['post_position_normalized'] = np.nan  # Initialize
        df.loc[valid_norm, 'post_position_normalized'] = (df.loc[valid_norm, 'post_position'] - 1) / (df.loc[valid_norm, 'num_horses_in_race'] - 1)
        # Fill NaNs (e.g., for single horse races or where post is missing) with middle value 0.5
        df['post_position_normalized'] = df['post_position_normalized'].fillna(0.5)

        df['is_inside_post'] = ((df['post_position'] == 1) & df['num_horses_in_race'].notna()).astype(int)
        df['is_outside_post'] = ((df['post_position'] == df['num_horses_in_race']) & df['post_position'].notna()).astype(int)

    # Convert distance to a standard unit (e.g., Furlongs)
    if 'distance' in df.columns and 'distance_unit' in df.columns:
        print("Standardizing distance to Furlongs...")
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        df['distance_unit'] = df['distance_unit'].astype(str).str.upper()

        def convert_to_furlongs(row):
            dist = row['distance']
            unit = row['distance_unit']
            if pd.isna(dist): return np.nan
            if unit == 'F': return dist  # Already furlongs
            if unit == 'Y': return dist / 220  # Yards to Furlongs
            if unit == 'M': return dist * 8  # Miles to Furlongs (approx)
            if unit == 'FT': return dist / 660  # Feet to Furlongs
            return np.nan  # Unknown unit

        df['distance_furlongs'] = df.apply(convert_to_furlongs, axis=1)
        median_dist = df['distance_furlongs'].median()
        if pd.isna(median_dist):  # If all values are NaN, use a default
            median_dist = 6.0  # Common race distance
            default_values_used['distance_furlongs'] = {'default_value': 6.0, 'reason': 'All values were NaN'}
            columns_with_defaults.append('distance_furlongs')
        df['distance_furlongs'] = df['distance_furlongs'].fillna(median_dist)
        print(f"Imputed missing 'distance_furlongs' with median: {median_dist:.2f}")
        # Drop original distance columns
        df = df.drop(columns=['distance', 'distance_unit'], errors='ignore')
    else:
        print("Warning: Missing distance information. Using default race distance.")
        df['distance_furlongs'] = 6.0  # Default common race distance
        default_values_used['distance_furlongs'] = {'default_value': 6.0, 'reason': 'Missing source columns'}
        columns_with_defaults.append('distance_furlongs')

    print("Categorizing horses based on attributes...")
    df = categorize_horses(df)  # Apply the categorization function

    # Create jockey-trainer combinations
    if 'jockey' in df.columns and 'trainer' in df.columns:
        print("Creating jockey-trainer combination features...")

        # Create combined identifier
        df['jockey_trainer'] = df['jockey'].astype(str) + '_' + df['trainer'].astype(str)
        df['jockey_trainer'] = df['jockey_trainer'].astype('category')

        # Use lookup table if available
        if use_jt_lookup and jt_lookup is not None:
            print("Applying historical jockey-trainer statistics from training data...")

            # Map win rates
            win_rates_dict = jt_lookup['win_rates']
            median_win_rate = jt_lookup['median_win_rate']
            df['jockey_trainer_win_rate'] = df['jockey_trainer'].map(win_rates_dict).fillna(median_win_rate)

            # Map in-the-money rates
            itm_rates_dict = jt_lookup['itm_rates']
            median_itm_rate = jt_lookup['median_itm_rate']
            df['jockey_trainer_itm_rate'] = df['jockey_trainer'].map(itm_rates_dict).fillna(median_itm_rate)

            # Create strength categories based on win rate using same bins as training script
            df['jt_strength'] = pd.cut(
                df['jockey_trainer_win_rate'],
                bins=[-0.001, 0.08, 0.15, 0.25, 1.01],
                labels=['Weak', 'Average', 'Strong', 'Elite'],
                right=False
            )

            print(f"Applied win rates to {sum(df['jockey_trainer'].isin(win_rates_dict.keys()))} entries")
            print(f"Used median win rate ({median_win_rate:.4f}) for {sum(~df['jockey_trainer'].isin(win_rates_dict.keys()))} entries")
        else:
            # Since we don't have historical data, use default values based on industry averages
            # IMPORTANT: These values should match the medians from the training dataset
            # For now, we'll use typical values, but ideally these would be extracted from training data
            df['jockey_trainer_win_rate'] = 0.15  # Default win rate
            df['jockey_trainer_itm_rate'] = 0.33  # Default in-the-money rate
            default_values_used['jockey_trainer_win_rate'] = {'default_value': 0.15, 'reason': 'Industry average, no historical data'}
            default_values_used['jockey_trainer_itm_rate'] = {'default_value': 0.33, 'reason': 'Industry average, no historical data'}
            columns_with_defaults.extend(['jockey_trainer_win_rate', 'jockey_trainer_itm_rate'])

            # Create jockey-trainer strength categories based on default values
            df['jt_strength'] = 'Average'  # Default category
            df['jt_strength'] = df['jt_strength'].astype('category')
            default_values_used['jt_strength'] = {'default_value': 'Average', 'reason': 'No historical data'}
            columns_with_defaults.append('jt_strength')

        print("Jockey-trainer features created successfully")
    else:
        print("Warning: Missing jockey or trainer data. Cannot create combination features.")

    # Drop unnecessary columns (adjust based on final feature set)
    columns_to_drop = [
        'program_num', 'comment', 'last_race_date',
        'last_race_track', 'track_name', 'post_time', 'weather',
        'medication', 'owner', 'track_condition',  # original track condition
        'post_position',  # original post position
        'horse_dob', 'race_type',  # original race type
        'last_race_time', 'second_call_time', 'final_time',  # Original time columns if not dropped earlier
        'win_time_seconds',  # Example: If win time isn't useful as predictor
        'num_past_wins', 'num_past_starts',  # Used for rates/categories, might be redundant
        'track_code',  # Explicitly dropping track_code
        'race_date'    # Explicitly dropping race_date
    ]
    # Add original time columns if they were converted and exist
    time_cols_original = ['win_time', 'finish_time', 'last_race_time', 'second_call_time', 'final_time']
    columns_to_drop.extend([col for col in time_cols_original if col in df.columns])

    print(f"Dropping unnecessary columns: {', '.join([c for c in columns_to_drop if c in df.columns])}")
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # Handle missing values robustly
    print("Handling missing values...")

    # Add tracking for NaN counts
    nan_summary = {}

    # Identify column types
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Remove race_id from feature columns if it exists
    if 'race_id' in numerical_cols:
        numerical_cols.remove('race_id')
    if 'race_id' in categorical_cols:
        categorical_cols.remove('race_id')

    # Remove original_horse_name from feature columns
    if 'original_horse_name' in numerical_cols:
        numerical_cols.remove('original_horse_name')
    if 'original_horse_name' in categorical_cols:
        categorical_cols.remove('original_horse_name')

    print(f"Identified {len(numerical_cols)} numerical columns.")
    print(f"Identified {len(categorical_cols)} categorical columns.")

    # Count total number of rows
    total_rows = len(df)
    print(f"Total number of rows in dataset: {total_rows}")

    print("\n--- NaN COUNTS BEFORE IMPUTATION ---")
    print("Column Name                   | NaN Count | % of Total Rows")
    print("-" * 65)

    # Gather and display NaN counts before imputation
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            pct = (nan_count / total_rows) * 100
            nan_summary[col] = nan_count
            print(f"{col:<30} | {nan_count:^9} | {pct:^9.2f}%")

    print("-" * 65)

    # Handle missing values in numerical columns
    for col in numerical_cols:
        # Count NaNs before imputation if we haven't already
        if col not in nan_summary:
            nan_count = df[col].isnull().sum()
            nan_summary[col] = nan_count
        else:
            nan_count = nan_summary[col]

        if nan_count > 0:
            if col == 'age' and pd.isna(df[col].median()):
                print(f"Warning: All {col} values are NaN. Using a default value of 5.")
                df[col] = df[col].fillna(5)  # Use a reasonable default value
                print(f"Imputed {nan_count} missing values ({(nan_count/total_rows)*100:.2f}% of rows) in '{col}' with fixed value 5")
                if col not in columns_with_defaults:
                    default_values_used[col] = {'default_value': 5, 'reason': 'All values were NaN'}
                    columns_with_defaults.append(col)
            else:
                median_val = df[col].median()
                if pd.isna(median_val):  # If all values are NaN, use a default
                    if col == 'age':
                        median_val = 5
                        if col not in columns_with_defaults:
                            default_values_used[col] = {'default_value': 5, 'reason': 'All values were NaN'}
                            columns_with_defaults.append(col)
                    elif col == 'days_since_last_race':
                        median_val = 30
                        if col not in columns_with_defaults:
                            default_values_used[col] = {'default_value': 30, 'reason': 'All values were NaN'}
                            columns_with_defaults.append(col)
                    elif col == 'distance_furlongs':
                        median_val = 6.0
                        if col not in columns_with_defaults:
                            default_values_used[col] = {'default_value': 6.0, 'reason': 'All values were NaN'}
                            columns_with_defaults.append(col)
                    else:
                        median_val = 0  # Generic default
                        if col not in columns_with_defaults:
                            default_values_used[col] = {'default_value': 0, 'reason': 'All values were NaN'}
                            columns_with_defaults.append(col)
                df[col] = df[col].fillna(median_val)
                print(f"Imputed {nan_count} missing values ({(nan_count/total_rows)*100:.2f}% of rows) in numerical column '{col}' with value {median_val:.2f}")

    # Handle missing values in categorical columns
    for col in categorical_cols:
        # Count NaNs before imputation if we haven't already
        if col not in nan_summary:
            nan_count = df[col].isnull().sum()
            nan_summary[col] = nan_count
        else:
            nan_count = nan_summary[col]

        if nan_count > 0 or (isinstance(df[col].dtype, pd.CategoricalDtype) and 'Unknown' not in df[col].cat.categories):
            # Convert to string to handle mixed types or add 'Unknown' category
            if not pd.api.types.is_string_dtype(df[col]) and not isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype(str)

            if isinstance(df[col].dtype, pd.CategoricalDtype):
                if 'Unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories('Unknown')
                df[col] = df[col].fillna('Unknown')
                print(f"Imputed {nan_count} missing values ({(nan_count/total_rows)*100:.2f}% of rows) in categorical column '{col}' with 'Unknown'")
            else:  # Handle object/string columns
                df[col] = df[col].fillna('Unknown')
                print(f"Imputed {nan_count} missing values ({(nan_count/total_rows)*100:.2f}% of rows) in object column '{col}' with 'Unknown'")

    # Limit high-cardinality categorical variables
    print("Limiting high-cardinality categorical variables...")
    MAX_CATEGORIES = 50  # Adjust as needed
    cols_to_limit = []
    for col in categorical_cols:
        # Use isinstance instead of is_categorical_dtype
        is_cat = isinstance(df[col].dtype, pd.CategoricalDtype)
        # Get number of unique values - convert to object temporarily if needed
        col_data = df[col].astype(object) if is_cat else df[col]
        n_unique = col_data.nunique()

        # Identify columns to limit (exclude manually created low-cardinality ones)
        if n_unique > MAX_CATEGORIES and col not in [
            'standard_condition', 'class_level', 'sex',
            'age_category', 'win_rate_category',
            'odds_category', 'experience_category',
            'is_inside_post', 'is_outside_post',  # These are binary/low cardinality
            'jt_strength'  # Our new jockey-trainer strength category
            ]:
            cols_to_limit.append(col)
            print(f"Identified high-cardinality column '{col}' with {n_unique} categories. Will limit to top {MAX_CATEGORIES}.")

    # Apply the limitation
    for col in cols_to_limit:
        # Get top categories
        top_categories = df[col].value_counts().nlargest(MAX_CATEGORIES).index.tolist()

        # Check if it's a categorical column with isinstance
        is_cat = isinstance(df[col].dtype, pd.CategoricalDtype)

        if is_cat:
            # For categorical columns, handle properly
            # First ensure 'Other' is a category
            if 'Other' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories('Other')

            # Create a temporary Series with the replacements
            temp_series = df[col].copy()
            temp_series = temp_series.apply(lambda x: x if (x in top_categories and not pd.isna(x)) else 'Other')

            # Ensure we maintain the categorical dtype when assigning back
            df[col] = pd.Categorical(temp_series, categories=df[col].cat.categories)
            # Remove unused categories after ensuring it's categorical
            df[col] = df[col].cat.remove_unused_categories()
        else:
            # For object/string columns, simpler approach
            df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
            # Convert to category after transformation
            df[col] = df[col].astype('category')

    # Re-identify categorical columns after changes to ensure consistency
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    # Remove race_id from categorical_cols again if it got re-added
    if 'race_id' in categorical_cols:
        categorical_cols.remove('race_id')
    if 'original_horse_name' in categorical_cols:
        categorical_cols.remove('original_horse_name')

    # Ensure all categorical columns are treated as 'category' type
    for col in categorical_cols:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype('category')

    # Final check for NaNs
    final_nan_sum = df.isnull().sum().sum()
    print(f"\nFinal check for NaNs in features: {final_nan_sum}")

    if final_nan_sum > 0:
        print("\n--- REMAINING NaN VALUES AFTER INITIAL IMPUTATION ---")
        print("Column Name                   | NaN Count | % of Total Rows")
        print("-" * 65)

        null_cols = df.isnull().sum()[df.isnull().sum() > 0]
        for col, count in null_cols.items():
            pct = (count / total_rows) * 100
            print(f"{col:<30} | {count:^9} | {pct:^9.2f}%")

        print("-" * 65)
        print("Attempting final fill with 0 for numeric and 'Unknown' for categorical columns...")

        for col in df.columns:
            if df[col].isnull().any():
                nan_count = df[col].isnull().sum()
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(0)
                    print(f"Final imputation: filled {nan_count} missing values in numeric column '{col}' with 0")
                else:
                    if isinstance(df[col].dtype, pd.CategoricalDtype):
                        if 'Unknown' not in df[col].cat.categories:
                            df[col] = df[col].cat.add_categories('Unknown')
                        df[col] = df[col].fillna('Unknown')
                    else:
                        df[col] = df[col].fillna('Unknown').astype('category')
                    print(f"Final imputation: filled {nan_count} missing values in categorical column '{col}' with 'Unknown'")

    # Print summary of imputation
    print("\n--- IMPUTATION SUMMARY ---")
    print(f"Total rows in dataset: {total_rows}")
    print(f"Total NaN values before imputation: {sum(nan_summary.values())}")
    print(f"Average NaN values per column with missing data: {sum(nan_summary.values())/len(nan_summary) if nan_summary else 0:.2f}")
    print(f"Maximum NaN values in a single column: {max(nan_summary.values()) if nan_summary else 0}")

    # Sort columns by NaN count and show the top 5 most affected
    if nan_summary:
        print("\nTop 5 columns with most missing data:")
        top_columns = sorted(nan_summary.items(), key=lambda x: x[1], reverse=True)[:5]
        for col, count in top_columns:
            pct = (count / total_rows) * 100
            print(f"  - {col}: {count} NaNs ({pct:.2f}% of rows)")

    # Final check after everything
    final_check = df.isnull().sum().sum()
    print(f"\nFinal NaN count after all imputations: {final_check}")
    if final_check > 0:
        print("WARNING: Some NaN values still remain in the dataset!")
    else:
        print("SUCCESS: All NaN values have been successfully imputed.")

    # Print summary of default values used
    print("\n--- DEFAULT VALUES SUMMARY ---")
    if columns_with_defaults:
        print(f"Total columns using default values: {len(columns_with_defaults)}")
        print("\nColumns with default values:")
        print("Column Name                   | Default Value     | Reason")
        print("-" * 80)

        for col in columns_with_defaults:
            if col in default_values_used:
                default_value = default_values_used[col]['default_value']
                reason = default_values_used[col]['reason']

                # Format the default value for display
                if isinstance(default_value, float):
                    default_str = f"{default_value:.2f}"
                else:
                    default_str = str(default_value)

                print(f"{col:<30} | {default_str:<18} | {reason}")

        print("-" * 80)
        print("\nNote: Default values were used when data was missing entirely or all values were NaN.")
        print("These defaults are based on industry standards or historical race data.")
    else:
        print("No default values were used - all values were derived from the data.")

    # Prepare data for model prediction
    print("Preparing data for model prediction...")

    # Make sure we only use features available for prediction
    X = df.drop(columns=['race_id', 'original_horse_name'], errors='ignore')

    # Convert categorical columns to codes for model fitting
    categorical_feature_names = []
    for col in X.columns:
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            categorical_feature_names.append(col)
            # Just convert to codes
            X[col] = X[col].cat.codes

    # Apply the preprocessor
    print("Applying preprocessor...")
    X_processed = preprocessor.transform(X)

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_processed)

    # Add predictions to the original dataframe
    df['predicted_finish'] = predictions

    # Create race-by-race predictions
    print("Generating race-by-race predictions...")

    # Create output directory if it doesn't exist
    output_dir = 'race_predictions'
    os.makedirs(output_dir, exist_ok=True)

    # Create CSV with all predictions
    predictions_df = df[['race_id', 'race_number', 'original_horse_name', 'predicted_finish']]
    if 'dollar_odds' in df.columns:
        predictions_df['odds'] = df['dollar_odds']
    if 'jockey' in df.columns:
        predictions_df['jockey'] = df['jockey']
    if 'trainer' in df.columns:
        predictions_df['trainer'] = df['trainer']

    # Save all predictions to a single CSV
    all_predictions_path = os.path.join(output_dir, 'all_race_predictions.csv')
    predictions_df.to_csv(all_predictions_path, index=False)
    print(f"All predictions saved to {all_predictions_path}")

    # Create individual race predictions and plots
    race_ids = df['race_id'].unique()

    # Create a combined HTML report
    html_report = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Race Predictions</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #003366; }
            h2 { color: #0055A4; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            tr:hover { background-color: #f5f5f5; }
            .top3 { background-color: #e6f7ff; }
            .winner { background-color: #d4edda; font-weight: bold; }
            .race-section { margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 20px; }
            .race-info { color: #666; margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <h1>Horse Race Predictions</h1>
    """

    # Process each race
    for race_id in race_ids:
        race_df = df[df['race_id'] == race_id].copy()
        race_num = race_df['race_number'].iloc[0] if 'race_number' in race_df.columns else 'Unknown'

        # Sort by predicted finish position
        race_df = race_df.sort_values('predicted_finish')

        # Create individual race CSV
        race_csv_path = os.path.join(output_dir, f'race_{race_num}_predictions.csv')
        race_predictions = race_df[['original_horse_name', 'predicted_finish']]
        if 'dollar_odds' in race_df.columns:
            race_predictions['odds'] = race_df['dollar_odds']
        if 'jockey' in race_df.columns:
            race_predictions['jockey'] = race_df['jockey']
        if 'trainer' in race_df.columns:
            race_predictions['trainer'] = race_df['trainer']

        # Save individual race predictions
        race_predictions.to_csv(race_csv_path, index=False)
        print(f"Race {race_num} predictions saved to {race_csv_path}")

        # Create visualization for this race
        plt.figure(figsize=(10, 6))
        race_chart = sns.barplot(x='original_horse_name', y='predicted_finish', data=race_df.sort_values('predicted_finish'))
        plt.title(f'Race {race_num} - Predicted Finish Positions')
        plt.ylabel('Predicted Finish Position (lower is better)')
        plt.xlabel('Horse Name')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save plot
        race_plot_path = os.path.join(output_dir, f'race_{race_num}_predictions.png')
        plt.savefig(race_plot_path)
        plt.close()
        print(f"Race {race_num} visualization saved to {race_plot_path}")

        # Add to HTML report
        html_report += f"""
        <div class="race-section">
            <h2>Race {race_num}</h2>
            <div class="race-info">
        """

        # Add race details if available
        if 'distance_furlongs' in race_df.columns:
            avg_distance = race_df['distance_furlongs'].mean()
            html_report += f"<p>Distance: {avg_distance:.1f} furlongs</p>"

        if 'surface' in race_df.columns:
            surface = race_df['surface'].iloc[0]
            html_report += f"<p>Surface: {surface}</p>"

        if 'standard_condition' in race_df.columns:
            condition = race_df['standard_condition'].iloc[0]
            html_report += f"<p>Track Condition: {condition}</p>"

        html_report += """
            </div>
            <table>
                <tr>
                    <th>Predicted Position</th>
                    <th>Horse Name</th>
        """

        # Add optional columns to table header
        if 'dollar_odds' in race_df.columns:
            html_report += "<th>Odds</th>"
        if 'jockey' in race_df.columns:
            html_report += "<th>Jockey</th>"
        if 'trainer' in race_df.columns:
            html_report += "<th>Trainer</th>"
        if 'age' in race_df.columns:
            html_report += "<th>Age</th>"
        if 'sex' in race_df.columns:
            html_report += "<th>Sex</th>"

        html_report += "</tr>"

        # Sort horses by predicted finish position
        sorted_horses = race_df.sort_values('predicted_finish')

        # Add each horse to the table
        for i, (_, horse) in enumerate(sorted_horses.iterrows()):
            position = i + 1

            # Determine row class based on position
            if position == 1:
                row_class = "winner"
            elif position <= 3:
                row_class = "top3"
            else:
                row_class = ""

            html_report += f"""
                <tr class="{row_class}">
                    <td>{position}</td>
                    <td>{horse['original_horse_name']}</td>
            """

            # Add optional columns to row
            if 'dollar_odds' in race_df.columns:
                odds = horse['dollar_odds'] if not pd.isna(horse['dollar_odds']) else "N/A"
                html_report += f"<td>{odds}</td>"
            if 'jockey' in race_df.columns:
                html_report += f"<td>{horse['jockey']}</td>"
            if 'trainer' in race_df.columns:
                html_report += f"<td>{horse['trainer']}</td>"
            if 'age' in race_df.columns:
                age = horse['age'] if not pd.isna(horse['age']) else "N/A"
                html_report += f"<td>{age}</td>"
            if 'sex' in race_df.columns:
                html_report += f"<td>{horse['sex']}</td>"

            html_report += "</tr>"

        html_report += """
            </table>
            <img src="{}" alt="Race {} Predictions" style="max-width: 100%; margin-top: 20px;">
        </div>
        """.format(f'race_{race_num}_predictions.png', race_num)

    # Finish HTML report
    html_report += """
    </body>
    </html>
    """

    # Save HTML report
    html_report_path = os.path.join(output_dir, 'race_predictions_report.html')
    with open(html_report_path, 'w') as f:
        f.write(html_report)
    print(f"HTML report saved to {html_report_path}")

    print("\nPrediction process complete!")
    print(f"All outputs saved to the '{output_dir}' directory")
    print(f"Open '{html_report_path}' in a web browser to view the complete report")

if __name__ == "__main__":
    main()