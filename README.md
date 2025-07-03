# clash_star_tracker
Clash Star Tracker helps players to track their team's performance using
image processing, optical character recognition, and automation of score
assignment.

To use the program within the cloned directory:
1. Provide your player names in players.txt with a new player name per line.
2. Provide any similar account names in multi_accounts.json with the alias as
   the key, and the list of accounts from strongest to weakest as the value.
3. Relocate any screenshots you'd like to process within an Images directory.
4. Run "python -m star_tracker" and the history should be written to
   player_history.csv.
5. If you'd like to view debugging data, run with --db

This program analyzes game screenshots by converting to HSL, (Hue, Saturation,
and Lightness) and detects changes in the minimum, maximum, and average
lightness of the image to locate features. This led me to develop powerful
functions like measure_image(), which works similarly to .measure in SPICE.
Given a specific threshold for the averaged stat of an image's row or column,
it will return the first and either the next or last occurrence of the stat
either rising or falling through this threshold. In this way its similar to
measuring propagation delay in circuit analysis by timing when the voltage
rises and falls through VDD/2! This threshold can either be absolute (from 0),
or relative  (how much it changed from the last measurement) Later I also added
the option for a secondary condition, as well as greater functionality to track
whenever two stats diverge and converge again within a given threshold.

When I realized that the proper threshold depends on the image itself and is
not a fixed constant, I developed sample_image() to perform adaptive
thresholding for any requested statistic. This function samples the
image by row or column, sorts from greatest to least, and returns the first
repeating value as a threshold. This way you can filter out the regular noise,
from the sharp jumps in minimum, or change in average. You can also provide it
with a global maximum to filter out, and provide an epsilon value for how
similar you want the repeating values to be.

Once the data has been located within the image, the image is preprocessed by
analyzing the darkness of the image, grouping the dark font outline with the
background of the image, and then inverting the colors so the font color is
black on a white background. Noise is further reduced by removing black shapes
in contact with the border, and shapes larger than a constant threshold,
however I found any further processing like convolutional filtering to reduce
clarity and the results of optical character recognition.

At first when OCR with pytesseract, I was getting mixed results and decided to
train my own model, however it was very inaccurate even after gathering, boxing
and processing over 200 player names. I found that further focusing my cropping
of features and smart use of page segmentation modes greatly improved my
accuracy than a custom model could. Player rank is especially important since
it is a great indicator of the difficulty of an attack, so these are isolated
from the player names and processed using PSM 10 for single characters, and
whitelisted so the response is only ever integers or integer-like characters.
When these occur, I translate them based on past experience via lookup table.
If no integer is located, I check to see if the player was seen before, then
assign the player the previously acknowledged rank. Otherwise I iterate through
previously seen players to assign it the next available entry in descending
order.

Player names are processed using PSM 7 for a single line of characters, and
autocorrect to a library of predefined character names for your team using
fuzzy matching at a fixed confidence. Enemy names are secondary in importance,
compared to enemy rank, so the first OCR result is simply added to a set, and
further OCR results are fuzzy matched to their initial readings.

I've added initial support for names that alias with each other via a provided
json file called multi_accounts.json. Inside is a dictionary where the key
is the OCR returned string, and the value is a list of all aliasing players in
descending rank. This ordering is consistent with war strength, but may drift
over time. This also changes depending on the aliased players participation
in each war, so manual adjustment can be made on a case by case basis.

The game rules are defined in data_structures.playerData.total_score() where
you can create your own custom bonuses or penalties for specific attacks. For
example, if you want to penalize dropping more than 10 ranks to attack a lower
player who hasn't been attacked yet: if there were no old stars you can reduce
the total score. Or if you want to reward players who jump up a certain amount
of ranks and score a new star, you can increase the total score.

The total scores are appended to a new column in the player_history csv, where
the final column is the total for all of the columns. The totals are then
printed to a leaderboard sorted by greatest stars to share with your team.

I plan to add an override for the feature recognition to accept hardcoded data
positions in the future in case of failure, however my current detection works
great for screenshots taken from the official clash of clans desktop app. I
also plan to create a UI and better support for implementing custom game rules.

Using FFT to perform edge detection may also be worth looking into, however I
am very satisfied with what I've learned throughout this project, and the tools
I've developed to realize my vision. I'd recommend running with --db to see all
the cool plots generated throughout the flow! If you have any suggestions feel
free to reach out to my email, or join my clan Another Land #8LYR2LLP
