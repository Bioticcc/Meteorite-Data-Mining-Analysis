# ------------------------------
# INITIAL DATA CLEANING SECTION
# ------------------------------

# This should just have what is currently in 01 but in a more modular form
# with function calls and important imports

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
# VISUALIZATION SECTION
# ------------------------------

# here we want to generate our various plots, which we will decide on in the notebook.
# should save plots to figure and table output folders, depending on what they are.
# we want lots of clustering related plots.