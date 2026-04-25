from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------
# INITIAL DATA CLEANING SECTION
# ------------------------------

# This should just have what is currently in 01 but in a more modular form
# with function calls and important imports

def load_raw_data():
    # First we load the dataset into the notebook in a pandas dataframe, and view the first 20 rows.
    df_original = pd.read_csv("../data/raw/Meteorite_Landings.csv")
    return df_original


def clean_missing_values(df_original):
    df = df_original.copy() # We make a copy of the original dataframe, so we can manipulate it without changing the original data.

    # Now that we have the dataframe, lets go through each row and column, and check for missing values.
    # Notabley, here face our first question. Do we just delete invalid rows, or give them a default value?
    # For now, I think we will give invalid/empty values a default of -1 for numeric, or N/A for string values.

    df = df.replace(r"^\s*$", np.nan, regex=True) # Replace any empty values with NaN.

    cols_str = df.select_dtypes(include=["object", "string"]).columns # Get all the string columns.
    cols_num = df.select_dtypes(include=["number"]).columns # Get all the numeric columns. (not used here but good to have)

    df[cols_str] = df[cols_str].fillna("N/A") # Fill all string columns with "N/A" for NaN values.

    return df


def convert_numeric_columns(df):
    # Now that we have imported the dataset and filled in the missing values, lets do a few checks:

    # 1. Set the numeric columns to numeric (in case some were saved as string but still are "123", etc)
    numeric_features = ["id", "mass (g)", "reclat", "reclong"]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce") # converts to numeric, NaN if invalid. will set to -1 again at the end.

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64") # converts year to integers.

    # 2. Check again for NaN values after our conversions, and fill with default vals if needed.
    changed_cols = ["id", "mass (g)", "reclat", "reclong"]
    print(df[changed_cols].isna().sum()) # checked for NaN values, was 0! so we can continue.

    # 3. Ensuring our coordinates and mass columns are within the correct ranges:
    invalid_lat = df["reclat"].notna() & ~df["reclat"].between(-90, 90)
    invalid_lon = df["reclong"].notna() & ~df["reclong"].between(-180, 180)

    print("Invalid latitude rows:", invalid_lat.sum())
    print("Invalid longitude rows:", invalid_lon.sum())

    bad_idx = df.index[invalid_lon]
    print(df.loc[bad_idx, ["id", "name", "reclat", "reclong", "GeoLocation"]])

    # Interesting! Notice we have a single invalid longitude value across the entire dataset. Its 354.47333,
    # for the meteor "Meridiani Planum", a meteor that was discovered on mars. Was this a naming mistake? Interesting.
    # I wont edit this row for now, since it is worth noting.

    return df


def remove_duplicates(df):
    # Next lets find and get rid of any duplicate rows, if any.
    duplicates_count = df.duplicated().sum()
    print("Number of duplicate rows:", duplicates_count)

    df = df.drop_duplicates().reset_index(drop=True) # Drop duplicate rows if any.

    print("Dataset size after removing dupliates:", len(df))
    print("Remaining duplicates:", df.duplicated().sum()) # Check again for duplicates, should be 0 now.

    # Well this cell was entirely pointless, as there was not a single duplicate. wow!
    # However, we should check for duplicate IDs specifically, since they are supposed to be unique.
    dup_id_rows = df.duplicated(subset=["id"], keep=False).sum()
    print("Rows with duplicate IDs:", dup_id_rows)

    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    print("Dataset size after removing dupliates:", len(df))
    print("Remaining duplicate IDs:", df.duplicated(subset=["id"]).sum())

    # Nope! no duplicate ID's!

    return df


def save_clean_data(df):
    # Finally, lets save our cleaned dataset to our processed data folder
    output_path = Path("../data/processed/meteorite_landings_clean.csv")
    df.to_csv(output_path, index=False)
    print("Saved clean data to:", output_path)
    return output_path


def run_data_cleaning():
    df_original = load_raw_data()
    df = clean_missing_values(df_original)
    df = convert_numeric_columns(df)
    df = remove_duplicates(df)
    save_clean_data(df)
    return df


# ------------------------------
# DATA PREPROCESSING SECTION
# ------------------------------

# log transform mass
# feature selection
# make the distance from equator feature
# make a region feature so we can group by continent / ocean
# encode fell vs found to 1 and 0
# subsets, fell only, fall only, etc
# normalize after we add everything

# make a note that for the fell vs found subsets, we can cluster them seperately
# and together, to show the possible biases due to population density in the region


# ------------------------------
# CLUSTERING AND EVALUATION SECTION
# ------------------------------

# K-means clustering:
# We find optimal K via silhouette score and elbow method (if we have time) 
# via looping through a range of K values, clustering, then evaluating the silhouette score for each K, and plotting the results to find the optimal K.
# we do this for each of our subsets, for visualization and research question purposes:
# - full dataset
# - fell only
# - found only
# we hsould find optimal k for each, then also do with the same k. 
# so at least right now we should have 6 different clusterings


# ------------------------------
# VISUALIZATION AND ANALYSIS SECTION
# ------------------------------

# here we want to generate our various plots, which we will decide on in the notebook.
# should save plots to figure and table output folders, depending on what they are.
# we want lots of clustering related plots.


# ------------------------------
# MAIN - runs everything in order
# ------------------------------

if __name__ == "__main__":
    df = run_data_cleaning()
