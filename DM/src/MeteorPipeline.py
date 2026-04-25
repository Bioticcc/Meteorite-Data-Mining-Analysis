from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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


def add_log_mass(df):
    # Now we log transform mass due to its absurd size.
    # Skew before was ~77, after log transform it drops to ~0.9 which is way better.
    df["log_mass"] = np.log1p(df["mass (g)"])
    return df


def add_fall_binary(df):
    # We also want to make fell vs found binary for clustering later.
    df["fall_binary"] = df["fall"].map({"Found": 0, "Fell": 1})
    return df


def add_continent_country(df):
    # Next, lets do some grouping! using geolocation, we split each meteorite into which continent it landed on.

    df["reclong_norm"] = ((df["reclong"] + 180) % 360) - 180  # normalize to [-180, 180]
    lat_ok = df["reclat"].between(-90, 90)
    lon_ok = df["reclong_norm"].between(-180, 180)
    not_missing = df["reclat"].notna() & df["reclong_norm"].notna()
    zero_zero = (df["reclat"] == 0) & (df["reclong_norm"] == 0)  # treat as placeholder

    valid = not_missing & lat_ok & lon_ok & ~zero_zero
    print("Valid geolocation rows:", int(valid.sum()))
    print("Invalid/unknown geolocation rows:", int((~valid).sum()))

    # create a GeoDataFrame with valid geolocations
    points = gpd.GeoDataFrame(
        df.loc[valid].copy(),
        geometry=gpd.points_from_xy(df.loc[valid, "reclong"], df.loc[valid, "reclat"]),
        crs="EPSG:4326" # converts it to coords
    )

    # This is the world polygons dataset we downloaded from natural earth, lets us bound points to land boxes
    # and determine country and continent!
    world = gpd.read_file("../data/external/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp").to_crs("EPSG:4326")
    land = world[["ADMIN", "CONTINENT", "geometry"]].rename(
        columns={"ADMIN": "country_land", "CONTINENT": "continent_land"}
    )

    # this is where geopandas checks which points meteorites are in.
    joined = gpd.sjoin(points, land, how="left", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")]  # border-safe

    # This is for the invalid coords
    df["country"] = "Unknown"
    df["continent"] = "Unknown"

    # valid coords but no land, so ocean
    df.loc[valid, "country"] = "Open Ocean"
    df.loc[valid, "continent"] = "Open Ocean"

    # and now we fill the valid coords if they have a country_land or continent_land!
    df.loc[joined.index, "country"] = joined["country_land"].fillna("Open Ocean")
    df.loc[joined.index, "continent"] = joined["continent_land"].fillna("Open Ocean")

    return df


def add_dist_equator(df):
    # before we save, also get the distance from the equator in kilometers.
    df["dist_equator_km"] = df["reclat"].abs() * 111.32
    return df


def save_processed_data(df):
    # Now we save our new and improved preprocessed dataset!
    output_path = Path("../data/processed/meteorite_landings_processed.csv")
    df.to_csv(output_path, index=False)
    print("Saved processed data to:", output_path)
    return output_path


def run_preprocessing(df):
    # make a note that for the fell vs found subsets, we can cluster them seperately
    # and together, to show the possible biases due to population density in the region
    df = add_log_mass(df)
    df = add_fall_binary(df)
    df = add_continent_country(df)
    df = add_dist_equator(df)
    save_processed_data(df)
    return df


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

def load_processed_data_for_clustering():
    # loading the dataset
    df = pd.read_csv("../data/processed/meteorite_landings_processed.csv")
    return df


def make_clustering_subsets(df):
    # subsets (CAN ADD MORE LATER, FOR NOW WE JUST USE THESE 3)
    subsets = {} # list of all of our subsets.
    subsets_continents = {}

    # main
    subsets["main"] = df.copy()

    # fell vs found
    subsets["fell"] = df[df["fall_binary"] == 1].copy()
    subsets["found"] = df[df["fall_binary"] == 0].copy()

    # all valid geo locations
    valid_geo = (
        df["reclat"].between(-90, 90)
        & df["reclong_norm"].between(-180, 180)
        & ~((df["reclat"] == 0) & (df["reclong_norm"] == 0))
    )
    subsets["geo_valid"] = df[valid_geo].copy()

    # land only
    land_mask = valid_geo & (~df["continent"].isin(["Open Ocean", "Unknown"]))
    subsets["land_only"] = df[land_mask].copy()

    # modern and historic for comparisons of bias?
    subsets["modern"] = df[df["year"] >= 1950].copy()
    subsets["historic"] = df[df["year"] < 1950].copy()

    for cont, n in df["continent"].value_counts().items(): # gets each continent and its total num records
        if cont in ["Open Ocean", "Unknown"]: # skips if non continent
            continue
        if n >= 500: # if continent has at least 500, it gets a subset!
            key = "cont_" + cont.lower().replace(" ", "_") # creates the key, cont_continentname
            subsets_continents[key] = df[df["continent"] == cont].copy() # adds it to subsets!

    print("Primary subsets:")
    for k, v in subsets.items():
        print(f"{k:20s} {v.shape}")

    print("\nContinent subsets:")
    for k, v in subsets_continents.items():
        print(f"{k:20s} {v.shape}")

    return subsets, subsets_continents


def get_clustering_feature_sets():
    # Now for the clustering! first, we want to identify our feature sets that we will use for clustering.
    feature_sets = {
        "baseline" : ["log_mass", "reclat", "reclong_norm", "year"],
        "time_space" : ["year", "reclat", "reclong_norm"],
        "mass_space" : ["log_mass", "reclat", "reclong_norm"],
        "full" : ["log_mass", "reclat", "reclong_norm", "year", "fall_binary"]
    }
    return feature_sets


def run_clustering_and_eval(subset, features, k_range=range(2, 11), n_init=20):
    # first we want only the selected features, and non empty rows or values.
    X = subset[features].apply(pd.to_numeric, errors='coerce').dropna() # converts to numeric and drops non numeric rows

    if len(X) < 3:
        raise ValueError("Not enough rows after dropping NaN's")

    # scale for k-means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rows = [] # each row
    models = {} # the k_means we have made so far, will get best at the end

    # valid K range
    valid_ks = [k for k in k_range if 2 <= k <= (len(X) - 1)]

    for k in valid_ks:
        kmeans = KMeans(n_clusters=k, n_init=n_init) # initializes the kmeans model
        labels = kmeans.fit_predict(X_scaled) # labels for each point

        if len(np.unique(labels)) < 2: # if only one cluster, silhouette score is not defined, skip
            continue

        rows.append({
            "k": k, # number of clusters
            "silhouette": silhouette_score(
                X_scaled,
                labels,
                sample_size=min(5000, len(X_scaled)),
                random_state=42
            ),
            "inertia": kmeans.inertia_ # cluster tightness
            })

        models[k] = kmeans # saves the clustering model

    # scores dataframe, sorted by silhouette score, best first
    scores = pd.DataFrame(rows).sort_values("silhouette", ascending=False)

    if scores.empty:
        raise ValueError("No valid silhouette results for this subset/features.")

    best_k = int(scores.iloc[0]["k"]) # gets our best performing K!
    best_model = models[best_k]

    clustered = subset.loc[X.index].copy()
    clustered["cluster"] = best_model.labels_

    return {
        "best_k": best_k,
        "scores": scores,           # all tested k results
        "model": best_model,        # fitted KMeans for best k
        "scaler": scaler,           # fitted scaler for this subset
        "clustered_df": clustered,  # rows used + cluster labels
        "feature_cols": features
    }


def run_primary_clustering(subsets, feature_sets):
    # next we call the clustering for each of our subsets and features sets.
    res_main_with_fall = run_clustering_and_eval(subsets["main"], feature_sets["full"]) # main with full featureset

    res_main = run_clustering_and_eval(subsets["main"], feature_sets["baseline"]) # main with baseline
    res_fell = run_clustering_and_eval(subsets["fell"], feature_sets["baseline"]) # fell with baseline
    res_found = run_clustering_and_eval(subsets["found"], feature_sets["baseline"]) # found with baseline

    res_geo_valid = run_clustering_and_eval(subsets["geo_valid"], feature_sets["baseline"]) # only valid geo with baseline
    res_land_only = run_clustering_and_eval(subsets["land_only"], feature_sets["baseline"]) # only land with baseline

    res_modern = run_clustering_and_eval(subsets["modern"], feature_sets["baseline"]) # only modern with baseline
    res_historic = run_clustering_and_eval(subsets["historic"], feature_sets["baseline"]) # only historic with baseline

    # our results!
    run_results = {
        "main_baseline": res_main,
        "main_with_fall": res_main_with_fall,
        "fell_baseline": res_fell,
        "found_baseline": res_found,
        "geo_valid_baseline": res_geo_valid,
        "land_only_baseline": res_land_only,
        "modern_baseline": res_modern,
        "historic_baseline": res_historic,
    }

    return run_results


def summarize_clustering_results(run_results):
    # summary of results sorted by score
    summary_rows = []
    for name, r in run_results.items():
        best = r["scores"].iloc[0]
        summary_rows.append({
            "run": name,
            "n_rows_clustered": len(r["clustered_df"]),
            "best_k": int(r["best_k"]),
            "best_silhouette": float(best["silhouette"]),
            "best_inertia": float(best["inertia"]),
            "features": ", ".join(r["feature_cols"]),
        })

    summary = pd.DataFrame(summary_rows).sort_values("best_silhouette", ascending=False)
    print("\nK-means summary:")
    print(summary)
    return summary


def build_cluster_centers(run_results):
    # cluster centers for each run transformed back to original feature space
    centers = {}
    for name, r in run_results.items():
        c = r["scaler"].inverse_transform(r["model"].cluster_centers_)
        centers[name] = pd.DataFrame(c, columns=r["feature_cols"])
        centers[name]["cluster"] = centers[name].index
    return centers


def save_clustering_outputs(summary, run_results, centers):
    # and save to file so we dont have to do this again!
    output_dir = Path("../outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "kmeans_summary.csv", index=False)

    for name, r in run_results.items():
        r["clustered_df"].to_csv(output_dir / f"{name}_clustered.csv", index=False)
        centers[name].to_csv(output_dir / f"{name}_centers.csv", index=False)


def run_clustering_pipeline():
    # Alright, now that we have a processed dataset, we start doing clustering and eval of said clusters.
    # This will be done on 3 subsets of the main dataset, full fell and found, with more features being
    # subset and clustered later.
    df = load_processed_data_for_clustering()
    subsets, subsets_continents = make_clustering_subsets(df)
    feature_sets = get_clustering_feature_sets()
    run_results = run_primary_clustering(subsets, feature_sets)
    summary = summarize_clustering_results(run_results)
    centers = build_cluster_centers(run_results)
    save_clustering_outputs(summary, run_results, centers)
    return df, subsets, subsets_continents, feature_sets, run_results, summary, centers


# ------------------------------
# VISUALIZATION AND ANALYSIS SECTION
# ------------------------------

# here we want to generate our various plots, which we will decide on in the notebook.
# should save plots to figure and table output folders, depending on what they are.
# we want lots of clustering related plots.

# for saving clusters:
BASE_FIG_DIR = Path("../outputs/figures/cluster plots")


def load_visualization_data():
    # Full cluster summary
    summary = pd.read_csv("../outputs/tables/kmeans_summary.csv")

    # row level data. each meteorite row and its assigned cluster. We use this for plotting
    clustered_data = {
        "main": pd.read_csv("../outputs/tables/main_baseline_clustered.csv"),
        "main_with_fall": pd.read_csv("../outputs/tables/main_with_fall_clustered.csv"),
        "fell": pd.read_csv("../outputs/tables/fell_baseline_clustered.csv"),
        "found": pd.read_csv("../outputs/tables/found_baseline_clustered.csv"),
        "geo_valid": pd.read_csv("../outputs/tables/geo_valid_baseline_clustered.csv"),
        "land_only": pd.read_csv("../outputs/tables/land_only_baseline_clustered.csv"),
        "modern": pd.read_csv("../outputs/tables/modern_baseline_clustered.csv"),
        "historic": pd.read_csv("../outputs/tables/historic_baseline_clustered.csv"),
    }

    # cluster level summarys, one row per cluster centroid, which we use for interpretation.
    # bar tables of typical cluster vals.
    centers_data = {
        "main": pd.read_csv("../outputs/tables/main_baseline_centers.csv"),
        "main_with_fall": pd.read_csv("../outputs/tables/main_with_fall_centers.csv"),
        "fell": pd.read_csv("../outputs/tables/fell_baseline_centers.csv"),
        "found": pd.read_csv("../outputs/tables/found_baseline_centers.csv"),
        "geo_valid": pd.read_csv("../outputs/tables/geo_valid_baseline_centers.csv"),
        "land_only": pd.read_csv("../outputs/tables/land_only_baseline_centers.csv"),
        "modern": pd.read_csv("../outputs/tables/modern_baseline_centers.csv"),
        "historic": pd.read_csv("../outputs/tables/historic_baseline_centers.csv"),
    }

    return summary, clustered_data, centers_data


def print_visualization_summary(summary):
    # highest quality kmeans runs
    summary_view = summary.sort_values("best_silhouette", ascending=False)[
        ["run", "best_k", "best_silhouette", "n_rows_clustered"]
    ]
    print("\nVisualization summary:")
    print(summary_view)


def plot_clusters_scatter(dfc, x_col, y_col, title, subdir="map"):
    # our generic cluster plotting function. Now we cluster!
    out_dir = BASE_FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=dfc, x=x_col, y=y_col,
        hue="cluster", palette="tab10", s=12, alpha=0.7, linewidth=0
    )
    plt.title(title)
    plt.tight_layout()

    filename = f"{title.lower().replace(' ', '_')}_{x_col}_vs_{y_col}.png"
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def print_center_tables(centers_data):
    # display the centroid tables (averages for each value for each cluster)
    for name in ["main", "fell", "found", "modern", "historic"]:
        print(f"\n{name}_centers:")
        print(centers_data[name])


def run_visualization_plots(clustered_data):
    # DISPLAY CLUSTERS! AND IT WORKS WOOOOOOOOAH
    # MAP: longitude vs latitude
    plot_clusters_scatter(clustered_data["main"], "reclong_norm", "reclat", "Main Clusters", subdir="map")
    plot_clusters_scatter(clustered_data["main_with_fall"], "reclong_norm", "reclat", "Main With Fall Clusters", subdir="map")
    plot_clusters_scatter(clustered_data["fell"], "reclong_norm", "reclat", "Fell Clusters", subdir="map")
    plot_clusters_scatter(clustered_data["found"], "reclong_norm", "reclat", "Found Clusters", subdir="map")
    plot_clusters_scatter(clustered_data["modern"], "reclong_norm", "reclat", "Modern Clusters", subdir="map")
    plot_clusters_scatter(clustered_data["historic"], "reclong_norm", "reclat", "Historic Clusters", subdir="map")

    # TIME-MASS: year vs log mass
    plot_clusters_scatter(clustered_data["main"], "year", "log_mass", "Main Time-Mass Clusters", subdir="time_mass")
    plot_clusters_scatter(clustered_data["main_with_fall"], "year", "log_mass", "Main With Fall Time-Mass Clusters", subdir="time_mass")
    plot_clusters_scatter(clustered_data["fell"], "year", "log_mass", "Fell Time-Mass Clusters", subdir="time_mass")
    plot_clusters_scatter(clustered_data["found"], "year", "log_mass", "Found Time-Mass Clusters", subdir="time_mass")
    plot_clusters_scatter(clustered_data["modern"], "year", "log_mass", "Modern Time-Mass Clusters", subdir="time_mass")
    plot_clusters_scatter(clustered_data["historic"], "year", "log_mass", "Historic Time-Mass Clusters", subdir="time_mass")


def run_visualization_pipeline():
    # Now that we have all the data loaded from clustering, lets do some visualization and eval!
    summary, clustered_data, centers_data = load_visualization_data()
    print_visualization_summary(summary)
    print_center_tables(centers_data)
    run_visualization_plots(clustered_data)
    return summary, clustered_data, centers_data


# ------------------------------
# MAIN - runs everything in order
# ------------------------------

if __name__ == "__main__":
    df = run_data_cleaning()
    df = run_preprocessing(df)
    run_clustering_pipeline()
    run_visualization_pipeline()
