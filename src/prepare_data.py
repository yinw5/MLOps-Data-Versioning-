"""Makes dataset v1 (raw) or v2 (cleaned + extra features), split into train/test.

    python ../src/prepare_data.py --version v1 --raw ../data/raw/athletes.csv
    python ../src/prepare_data.py --version v2 --raw ../data/raw/athletes.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA_DIR, LIFTS, RANDOM_STATE, TARGET, TEST_PATH, TEST_SIZE, TRAIN_PATH

HOWLONG_YEARS = {
    "Less than 6 months": 0.25,
    "6-12 months": 0.75,
    "1-2 years": 1.5,
    "2-4 years": 3.0,
    "4+ years": 5.0,
}


def add_total_lift(data):
    """add total_lift (sum of the 4 lifts), rows missing any lift get dropped"""
    data = data.copy()
    data[TARGET] = data[LIFTS].sum(axis=1, min_count=len(LIFTS))
    return data.dropna(subset=[TARGET])


def clean_v2(data):
    """cleaning code from the assignment"""
    # Remove not relevant columns
    data = data.dropna(subset=['region', 'age', 'weight', 'height', 'howlong', 'gender', 'eat',
                              'train', 'background', 'experience', 'schedule', 'howlong',
                              'deadlift', 'candj', 'snatch', 'backsq', 'experience',
                              'background', 'schedule', 'howlong'])
    data = data.drop(columns=['affiliate', 'team', 'name', 'athlete_id', 'fran', 'helen', 'grace',
                              'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups', 'train'])

    # Remove Outliers
    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]

    data = data[(data['deadlift'] > 0) & (data['deadlift'] <= 1105) | ((data['gender'] == 'Female')
                & (data['deadlift'] <= 636))]
    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]

    # Clean Survey Data
    decline_dict = {'Decline to answer|': np.nan}
    data = data.replace(decline_dict)
    data = data.dropna(subset=['background', 'experience', 'schedule', 'howlong', 'eat'])
    return data


def max_howlong_years(answer):
    """howlong -> years. some people picked more than one answer, take the longest"""
    years = [HOWLONG_YEARS[a] for a in str(answer).split("|") if a in HOWLONG_YEARS]
    return max(years) if years else np.nan


def add_features_v2(data):
    """extra features for v2: bmi, years of crossfit, a few yes/no survey flags"""
    data = data.copy()
    data["bmi"] = 703 * data["weight"] / data["height"] ** 2
    data["howlong_years"] = data["howlong"].map(max_howlong_years)
    flags = {
        "bg_college_sports": ("background", "I played college sports"),
        "bg_no_athletic": ("background", "I have no athletic background"),
        "exp_started_with_coach": ("experience", "with a coach"),
        "exp_level1_cert": ("experience", "Level 1 certificate"),
        "exp_trains_others": ("experience", "I train other people"),
        "sched_multi_per_day": ("schedule", "multiple workouts in a day"),
        "eat_weigh_measure": ("eat", "I weigh and measure"),
        "eat_paleo": ("eat", "Paleo"),
    }
    for name, (column, phrase) in flags.items():
        data[name] = data[column].str.contains(phrase, regex=False).astype(int)
    return data


def main():
    """main"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], required=True)
    parser.add_argument("--raw", required=True, help="path to the raw athletes.csv")
    args = parser.parse_args()

    data = pd.read_csv(args.raw, low_memory=False)
    print(f"raw rows: {len(data):,}")
    if args.version == "v2":
        data = add_features_v2(clean_v2(data))
    else:
        # not features anyway, and I don't want athlete names in a public repo
        data = data.drop(columns=["name", "athlete_id"])
    data = add_total_lift(data)
    print(f"{args.version} rows with total_lift: {len(data):,}  columns: {data.shape[1]}")

    # same seed + ratio for v1 and v2
    train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"wrote {TRAIN_PATH} ({len(train):,}) and {TEST_PATH} ({len(test):,})")


if __name__ == "__main__":
    main()
