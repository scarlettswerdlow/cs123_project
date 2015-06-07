###############################################################################
#                                                                             #
#  CMSC 12300                                                                 #
#  Project: Yelp Dataset Challenge                                            #
#  Code to approximate network from association rules analysis                #
#  Coded by Scarlett Swerdlow                                                 #
#  scarlettswerdlow@uchicago.edu                                              #
#  June 3, 2015                                                               #
#                                                                             #
###############################################################################

source("network/network_helpers.R")
data <- loadAndCleanData(EDGES_FN, BIZ_FN)
yelp <- data$yelp
biz <- data$biz

#################
#               #
#  EXPLORATORY  #
#               #
#################

# 25 business pairs with highest lift
top_lift <- getTopLift(yelp, 25, output=T, "network/top_lift.csv")
# Why are they all in Scotland?

# Business pairs with highest lift for each state
top_lift_state <- getTopLiftStateHelper(yelp, "biz_a_state", output=T, 
                                        "network/top_lift_by_state.csv")

# Interstate business pairs with highest lift for each state pair
top_lift_inter <- getTopLiftInterHelper(yelp, output=T,
                                        "network/top_interstate_lift_by_state_pair.csv")

#############
#           #
#  NETWORK  #
#           #
#############

g <- plotGraph(yelp) # Plot network graph

d <- getDegrees(g, output=T, "network/top_biz_by_degrees.csv") # Degrees

b <- getBetweenness(g, output=T, "network/top_biz_by_betweenness.csv") # Between
