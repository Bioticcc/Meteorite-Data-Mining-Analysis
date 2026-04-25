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
    As you can see, our skew is still somewhat bad, as the closer we are too 1, the worse, with +1 being agregious. However, I think its important to keep slight right skew if that tells the truth of the data, as it is certainly true that meteorites get absurdley large.