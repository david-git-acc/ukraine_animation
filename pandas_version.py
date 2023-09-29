from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import datetime as dt
import pandas as pd

from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors


#sources of the list of Ukrainian cities come from https://cityclock.org/

'''
A program which displays the relative search popularity of Ukrainian cities on a map of Ukraine, from the 20th of February 2022 to 25th of July, 2023
Creates an animated video, with every 2 frames corresponding to a single day. 
Each noted Ukrainian city on the map will have a circle to show its location, whose sizes denote how frequently searched they were on that day.
The colour of each circle denotes the pre-war population.

The search data for the 340 cities was pulled manually from Google trends. However, because each location would have a maximum value of 100 as Trends does not
show absolute search volume, I found the search volume for each city when they first reached 100 popularity. This maximum search volume for a given city would be 
multiplied to every data point about the city from 20/02/2022 to 25/07/2023, acting as a scalar multiplier to help compare between cities. 

'''

# This is the interpolation extent. A p-value of 4 means that there will be 4x as many frames as there were originally, or 3 interpolated frames generated
# for every original frame
p=4

the_20th_feb = dt.date(2022,2,20)

# Open all the data about the cities - each city will have a name, latitude-longitude coordinates,
# the scalar multiplier and its pre-war population
df = pd.read_csv("ukr_cities.csv")
# Turning into dict form for the program ( edited the program to use pandas instead)
# Stores all of the above data in a dictionary (called towns instead of cities)
towns_dict =  df.set_index("location").apply(lambda x: tuple([float(y) for y in x ]), axis=1).to_dict()


# This function opens the csv for a given city (e.g kiev.csv) (always lower case)
# Then it stores as a list of tuples each time (key) with the search popularity at that time (value)
def get_csv_info(file): 


    # Ignore the first 3 lines, they're not data
    # Keep a counter to ignore the first 3
    counter = 0

    # Parse the CSV with required parameters, all <1 values will be replaced with 1
    df = pd.read_csv("citydata/" + file + ".csv", skiprows=3, names=["time","size"],parse_dates=["time"],na_values=["<1"] )
    
    # Fill NA values and parse as int so we can perform numerical operations
    df = df.fillna("1")
    df["size"] = df["size"].astype(int)


    # This is the final product
    # This is a complicated way to basically just wrap each date-size pair into a list of tuples
    # set time as the index to be the keys to a dict, then flip the orientation of the frame, convert to dict,
    # unpack the "size" and then list(df.items()) to get the result
    info = list( df.set_index("time").apply(lambda x:x, axis=1).to_dict()["size"].items() )
    
    return info

# Same as above, but because the data is only stored for every week instead of every day, need to use
# interpolation to estimate the search frequency values for each DAY
def get_csv_interpolated_info(file):
    
    # Store the data here - we'll have an interpolation ratio of 6p:1
    new_info = []
    
    # This gives us the uninterpolated, weekly data - this is what we'll interpolate
    the_csv_info = get_csv_info(file)
    
    length = len(the_csv_info)
    
    # Interpolation process
    for i in range(0, length-1):
        
        # Get the data for this week and next week - we'll use these as start and endpoints 
        this_week = the_csv_info[i]
        next_week = the_csv_info[i+1]
        
        # These are the times: a = the date this week, b = the date next week
        a = this_week[0]
        b = next_week[0]
        
        # f_a and f_b are the search values at these times
        f_a = this_week[1]
        f_b = next_week[1]
             
        # This is where we'll store the interpolated data
        interpolated_values = []
        
        # The number of frames is determined by the p-value
        for j in range(1,7*p-1+1):

            # Using the linear interpolation formula, this is maths
            # If the p-value is >1, then we will need to adjust the day accordingly, not just 1 frame = 1 day
            interpolated_value = (a + dt.timedelta(days=int(j/p)), f_a + (f_b-f_a) * (j/p) / 7)
                
            interpolated_values.append(interpolated_value)
            
        # Don't forget to add the original uninterpolated data point
        new_info.append(this_week)
        
        # Now that we have all the interpolated data, add it to the list
        new_info += interpolated_values
        

        
    # Need to remember to add the data point at the end
    new_info.append(the_csv_info[length-1])
    
    return new_info
        
        
# Same as above but standardise the data to be up to 1000 from 100, then remove decimals
# I thought 1000 would be a good maximum while maintaining most precision (since the interpolated values
# would often be decimals)
def get_standardised_interpolated_data(string):
    
    return [(datetime, int( score*10) ) for datetime,score in get_csv_interpolated_info(string)]
        

# Same as above but we only care about the search frequency, not the dates
def get_standardised_interpolated_datenumber_data(placename):
    

    data = get_standardised_interpolated_data(placename)
    
    return [score for date,score in data]




'''
how I did this:

step 1: for each town, download the google trends data from 20th feb 2022 to 25th july 2023 for it, 
store this data as (townname).csv in this location

step 2: again for each town, get the number of search results when the trends data is at 100

step 3: for each town, set all_time_town_data[townname] = [x[1] for x in this_town_data] using the
GSIDD function

'''


# This dictionary will store dates as the key, and then an important event that happened on that date (value),
# this is used for context to explain why a city may suddenly become very large at a point in time
recent_events = {}

# Extracting the recent events data from our eventdata.txt file
with open("eventdata.txt","r") as event_file:
    # Each line is an event
    for line in event_file:
        # The date is stored in the first 10 characters, separated by /
        time = [int(x) for x in line[0:9+1].split("/")]
        # Convert from list of integers into a date
        time_as_date = dt.date(*time[::-1])
        # The information itself - kill the newlines
        info = line[11:].replace("\n","")
        # Add the information to the recent events dict
        recent_events.update({time_as_date : dt.datetime.strftime(time_as_date,'%d/%m/%Y') + ": " + info + "\n"})

# This constant determines the max number of characters which will be written to the screen before
# a newline is added, to prevent it from bleeding over into the plot
# 38 was the max I could have before the text started to bleed into the plot itself
newline_constant = 38

# This loop checks each event string and determines when to add newlines to prevent text bleeding
for time,event in recent_events.items():

    # We'll store the list of words here for each eventstring
    event_words = []

    # We'll build the current word from the text
    current_word = ""
    
    sentence_as_list = list(event)
    
    for char in sentence_as_list:

        current_word += char
        
        # If we find a space or /, then end the current word and begin building the nexto ne
        if char == " " or char == "/":
            event_words.append(current_word + " ")
            current_word = ""
        
    event_words.append(current_word)
    
    # Counter to check how many characters have elapsed
    counter = 0
    
    # Iterate over each word in the eventstring
    for ind, word in enumerate(event_words):
        
        # Iterate over each character in each word
        for ind2, char in enumerate(list(word)):
        
            counter += 1
            
            # If we reach the newline constant, add the newline
            if counter == newline_constant:
                # Check if we add the newline before this word or after, by checking to see if it's the last
                # character
                if ind2 == len(list(word)) - 1:
                    event_words[ind] += "\n"
                else:
                    # If it's not the last character, then add it to the previous word instead and 
                    # reset the counter
                    event_words[ind-1] += "\n"
                counter = 0
            
    # Once the newlines have been added, then we can reunify the string and replace the original event string
    # with this new one, to prevent text bleeding over into the plot
    recent_events[time] = "".join(event_words)
    
# Getting all the interpolated search data for each town
all_time_town_data = {  town : get_standardised_interpolated_datenumber_data(town.lower())  
                        for town in towns_dict.keys() }


# To prevent some cities from always being at the top and allow for a better viewing experience,
# we can subtract the minimum value from each location, so that every location will have at least 1
# point in time where it has no search popularity at all
for key,value in all_time_town_data.items():
    
    min_value = min(value)
    all_time_town_data[key] = [ x - min_value for x in value ] 



# This is the time to live (ttl) constant for each "recent event" - checks how many more frames an event
# should be displayed for before being removed to make way for the next recent events
# Since more frames (but same frame rate) from interpolation would make the ttl effectively lower, 
# multiply by p to counter
ttl_constant = 100*p



# crucial info
px=1/96
towns = list(towns_dict.keys())
coords = list( towns_dict.values() )
lats = np.asarray([x[0] for x in coords])
lons = np.asarray([x[1] for x in coords])
sizes = np.asarray([x[2] for x in coords])
popsizes =  np.cbrt( np.asarray( [x[3] for x in coords]) ) 

# Making the map
ukrmap = Basemap(llcrnrlat = 44, 
                 llcrnrlon= 22,
                 urcrnrlat=53, 
                 urcrnrlon=41,
                 resolution="i",
                 projection="merc",
                 epsg=6384)
                # epsg for Ukraine region, so we can display the satellite map

# Converting lats and lons into the corresponding x-y coordinates on the Basemap
xy_lons, xy_lats= ukrmap(lons, lats)

# Drawing the Ukrainian border
ukrmap.drawcountries(linewidth=1.25, )

# Satellite map overlay - I thought 1300 xpixels would be detailed enough
# Didn't want viewers to be staring at a black and white blank map
ukrmap.arcgisimage(service='World_Imagery', xpixels=1300)

# This is the colourmap that will be used to colour city circles, based on pre-war population
my_cmap=mcolors.LinearSegmentedColormap.from_list('rg',["darkred", "yellow", "lime"], N=256) 

# Initial scatter - becaause it's animated, need to instantiate everything initially
scattering = ukrmap.scatter(xy_lons, xy_lats, marker="o", edgecolors="black" , s=0, c=popsizes, cmap=my_cmap)

# The figure and axes are only implicitly generated with a Basemap instance, so we need to get hold of them
fig = plt.gcf()
ax = plt.gca()

# Ditto, need to manually set the dimensions
fig.set_figheight(1080*px)
fig.set_figwidth(1920*px)

# Initialising the colourbar
bar = plt.colorbar(mappable=scattering, extend="both", label="Pre-war population (Coded: f(x) = ³√(x))")

# This is where we will store the annotations - when a city becomes big enough 
# we will display its name on its circle
annotations = []

# Showing the most popular city: initialisation
maximum_city_text = plt.text(0.9, -0.075, f"Most popular city: Kiev",fontsize=15, ha="left", va = "bottom", transform=ax.transAxes)

events_text = plt.text(-0.25,0.975, 
                       "Recent events", 
                       fontsize=15, 
                       transform=ax.transAxes )


current_events = []
recent_events_texts = []

# Cumulative newline count - need to keep track of this so our recent events texts don't overlap
cum_newline_count = 0

def animate(i):
    
    global cum_newline_count
    global recent_events_texts
    global current_events
    global maximum_city_text
    global annotations
    global scattering
    
    the_date = the_20th_feb + dt.timedelta(days=int(i/p))

    # Need to remove all the annotations from the previous frame so we can update their sizes
    # for this frame - annotation size ~ circle size
    for annotation in annotations:
        annotation.remove()
            
    annotations = []

    # Clear all from the previous slide
    scattering.remove()
    maximum_city_text.remove()
    
    
    # Updating the current events
    for ind, value in enumerate( current_events ):
        # Decrement the time to live for each event
        current_events[ind][1] -= 1
        
        # If any event's ttl < 0, then we need to remove it
        if current_events[ind][1] <= 0:
            current_events.remove(current_events[ind])
            
            # Remove all the events so we can reinitialise them
            # Removing an event means we now have to change the positions of every other event
            # (This is not actually necessary now that all the oldest events are at the bottom, 
            # but I'm too lazy to remove this code since it will never be used again)
            for event in recent_events_texts:
                event.remove()
                
            recent_events_texts = []
            
            # Since there are now no events, our cumulative newline count is also now at 0
            cum_newline_count = 0
            
            for event_data in current_events:
                event = event_data[0]
                newline_count = event.count("\n")
                cum_newline_count += newline_count
                                                # This is a good approximation of the height of one line
                event_text = plt.text(-0.25, 0.975 -(1+cum_newline_count)/45,
                                        event,
                                        fontsize=9,  # Generating each event's text again
                                        transform=ax.transAxes,
                                        color="black",
                                        alpha=1,)
                recent_events_texts.append(event_text)
                

            

    # Check if there's an event today
    possible_event_today = recent_events.get(the_date)
    
    
    if possible_event_today and possible_event_today not in [x[0] for x in current_events]:
        # Pre-pend the new event, since the most recent events go at the top
        current_events = [[possible_event_today,ttl_constant]] + current_events
        
        # Same idea as above, kill all of the events, then reinitialise
        # This time, it's NECESSARY since as the new event goes at the top we need to shift downwards
        # the positions of all the current events
        for event in recent_events_texts:
            event.remove()
                
                
        # See above for explanations
        recent_events_texts = []
        
        cum_newline_count = 0
        
        for event_data in current_events:
            event = event_data[0]
            newline_count = event.count("\n")
            cum_newline_count += newline_count
            event_text = plt.text(-0.25, 0.975 -(1+cum_newline_count)/45,
                                    event,
                                    fontsize=9,
                                    transform=ax.transAxes,
                                    color="black",
                                    alpha=1,)
            recent_events_texts.append(event_text)
    

    
        
        
        



    
    ########################################################################################

    plt.title(f"War in Ukraine: most searched places on {dt.datetime.strftime(the_date,'%d/%m/%Y')}", fontsize=24)

    # This data stores the sizes of each city's circle at the given point in time
    # 64/1000 was a good scale factor to prevent them from becoming too large, 
    # don't forget that we need to multiply by the scale factor for each town (information[2])
    size_data = [ ( ( information[2] * all_time_town_data[town][i]  ) * 64/1000 )  
                    for town, information in towns_dict.items()] 
    
    # Need to determine most popular city at this given frame
    maximum = max(size_data)
    maximum_index = size_data.index(maximum)
    maximum_city = towns[maximum_index]

    # Annotating each sufficiently large city/town
    for index, town in enumerate( towns ):
        # Only give the label if t
        if size_data[index] > 200:
            # Since area of the circle grows linearly (not quadratically), need to do the same here
            fontsize = size_data[index] ** 0.5 / 3
            # Magic numbers ensure that the annotations stay inside the circle, found with trial and error
            annotations.append( ax.annotate(town, xy= (xy_lons[index] - 840* (size_data[index] ** 0.5 ) , xy_lats[index]), fontsize= fontsize * 5/ len(town) ) )

    maximum_city_text = plt.text(0.375, -0.07, 
                                 f"Most popular city: {maximum_city}",
                                 fontsize=20, 
                                 ha="left", 
                                 va = "bottom", 
                                 transform=ax.transAxes, 
                                 animated=True,)
    
    # Showing the circles on this frame
    scattering = ukrmap.scatter(xy_lons, xy_lats, 
                                marker="o", 
                                edgecolors="black" , 
                                s=size_data, 
                                c=popsizes, 
                                cmap=my_cmap)

    # Need to set an alpha so we can still see areas of overlap
    scattering.set_alpha(0.25)

# Animation
anim = FuncAnimation(fig, animate, frames = 512*p - (p-1), interval = 200/p)

# Save the animation
#anim.save('testing_morefps3.gif', fps=20)

plt.show()


    


