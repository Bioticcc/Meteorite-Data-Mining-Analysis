# Report Notes
Take note of any changes made here, as we will be using THIS to write the report as we continue 

## First commit - 4/18
We began by setting up the project structure, with all the folders and files that we hoped we would need. We planned out a project outline, and prepared our idea of a pipeline to work through. We have 4 jupyter notebooks for practicing/making decisions and testing with, then corresponding python files that we would then move our code into once its finalized. This split is because of our desire to turn this into a website later on, so we need something more solid then notebooks for the purposes of website design and UI. Also, having solid .py scripts makes for easier running of the full pipeline later, as well as orginization.

## Data Cleaning - 4/18
    We opened the dataset, and did the following:
    1. Checked for missing values replacing them with NaN
    2. Replaced NaN with -1 for numeric cols and N/A for string cols
    3. Checked that the numeric cols had ONLY numeric values, turned them into numeric just in case
    4. Did the same for year, turning it into an integer.
    5. searched for invalid latitude and longitude ranges, interestingly we found one! The meteor Meridiani Planum had a longitude of 350+ which is odd. The meteor was also discovered on MARS, adding to the confusion.
    6. Searched for duplicate rows, found none, deleted them anyway, dataset size was predictably unchanged.
    7. Searched for duplicate rows but specifically with duplicate ID's, since they should be unique. Also 0.
    8. We then exported the cleaned csv to our processed data folder.

## EDA - 4/25
    Here we got some interesting information! playing around in the jupyter notebook,
    I first tried skew tests, to see how agregious the number skews might be on select numerical columns. Now, i chose mass, reclat, and reclong, thinking that while mass was obvious, we might see something neat with the geolocation data. However, it only occured to me I may have misunderstood what exactly skew tells us. Since lat and long is bounded, the skew will never really be that egregious either way. Mass however, was ridiculously skewed at mass (g) = 77.020563. So next, we did a log transform on the data.
    After running transform, our skew changed like so:
    Skew before: 76.91011731918955
    Skew after : 0.9072237919413435 
    As you can see, our skew is still somewhat bad, as the closer we are too 1, the worse, with +1 being agregious. However, I think its important to keep slight right skew if that tells the truth of the data, as it is certainly true that meteorites get absurdely large.

    Then, we downloaded a dataset for earth polygons that we ccould then match our meteorite geo locations to! now we have some new features, with continent and country listed now, for our later research questions. It will be very interesting to see if there are certain continents with the most meteorites! We also added distance from the equator in kilometers as a feature

## Clustering - 4/25
    This is the big one! clusterin will be done with K-means with K determined via
    looping through a set range of possible K and computing the silhoette score on each.
    
    After subsetting, we see an interesting thing. fell is significantly lower then found! (reminder that fell is when people or sensors observe its passage through the sky to earth). It will be interesting to see the visual data for this, as well as by country and continent. 

    Primary subsets:
    main                 (45716, 16)
    fell                 (1107, 16)
    found                (44609, 16)
    geo_valid            (32187, 16)
    land_only            (32093, 16)
    modern               (43740, 16)
    historic             (1685, 16)

    Continent subsets:
    cont_antarctica      (22099, 16)
    cont_asia            (3562, 16)
    cont_africa          (2845, 16)
    cont_north_america   (1822, 16)
    cont_oceania         (648, 16)
    cont_europe          (617, 16)
    cont_south_america   (500, 16)

    We got quite a few other subsets as well, and can now start to look through them!

    # now that we subsets, we need to make feature sets for what we want to cluster things with. We also need to make a function that goes through a subset and featureset, runs kmeans clustering on a range of ks, then gets silhouettes scores for each as we go, then return a set of results with which K the subset used as the best one, the Kmeans model itself, its score, and the subset and feature.

    After selecting feature sets for each subset, we tried running our clustering function, however we ran into a rahter significant problem. Currently, we tried full silhouette score on each k and since its O(n^2) the larger the datasets the longer it takes, we genuinely could not get it to run for our full datasets for the life of us.
    To combat this, we switched to silhouette with a set sample size, making it much smaller, and (hopefully) without much loss.

    Once we got our clusters, we made a summary table, and saved all the cluster information to the outputs/tables as various csvs before we use them in visualization for analysis.