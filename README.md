# ukraine_animation

A program which displays the relative search popularity of Ukrainian cities on a map of Ukraine, from the 20th of February 2022 to 25th of July, 2023 Creates an animated video, with every 2 frames corresponding to a single day. Each noted Ukrainian city on the map will have a circle to show its location, whose sizes denote how frequently searched they were on that day. The colour of each circle denotes the pre-war population.

The search data for the 340 cities was pulled manually from Google trends. However, because each location would have a maximum value of 100 as Trends does not show absolute search volume, I found the search volume for each city when they first reached 100 popularity. This maximum search volume for a given city would be multiplied to every data point about the city from 20/02/2022 to 25/07/2023, acting as a scalar multiplier to help compare between cities.

Since I did not know how to use Pandas when this was made (this was reuploaded due to a problem with pushing it from my computer), there is no use of Pandas in the program. I regret not having 
learnt it sooner than when I did.
