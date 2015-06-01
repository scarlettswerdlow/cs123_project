###############################################################################
#                                                                             #
#  CMSC 12300                                                                 #
#  Project: Yelp Dataset Challenge                                            #
#  Code to approximate network from association rules analysis                #
#  Coded by Scarlett Swerdlow                                                 #
#  scarlettswerdlow@uchicago.edu                                              #
#  May 31, 2015                                                               #
#                                                                             #
###############################################################################

setwd("~/Google Drive/Grad school/Courses/CMSC123/project") # To be set by user

##################
#                #
#  DEPENDENCIES  #
#                #
##################

library(data.table)
library(igraph)
library(jsonlite)

##########
#        #
#  DATA  #
#        #
##########

yelp <- fread("network/yelp_network.csv", header=T) # From a-rules analysis
yelpdf <- as.data.frame(yelp)
names(yelpdf) <- c("row", "biz_a", "biz_b", "prob_a", "prob_b", "prob_ab", "conf", "lift")

# Add some business data
biz <- stream_in(file("yelp_academic_dataset_business.json"), flatten=T)
yelpdf$biz_a_name <- biz$name[match(yelpdf$biz_a, biz$business_id)]
yelpdf$biz_b_name <- biz$name[match(yelpdf$biz_b, biz$business_id)]
yelpdf$biz_a_state <- biz$state[match(yelpdf$biz_a, biz$business_id)]
yelpdf$biz_b_state <- biz$state[match(yelpdf$biz_b, biz$business_id)]

#################
#               #
#  EXPLORATORY  #
#               #
#################

# 25 business pairs with highest lift
top_lift <- tail(yelpdf[order(yelpdf$lift),], 25)[,-1]
write.csv(top_lift, "network/top_lift.csv")
# Why are they all in Scotland?

# Business pairs with highest lift for each state
top_lift_state <- data.frame()
for (state in unique(biz$state)) {
  x <- yelpdf[yelpdf$biz_a_state == state & yelpdf$biz_b_state == state,]
  if (nrow(x) == 0) { next }
  top <- x[which.max(x$lift),]
  top$state <- state
  top_lift_state <- rbind(top_lift_state, top)
}
write.csv(top_lift_state, "network/top_lift_by_state.csv")

# Interstate business pairs with highest lift for each state pair
lift_inter <- data.frame()
c <- combn(unique(biz$state),2)
for (i in 1:ncol(c)) {
  a <- c[1,i][[1]]
  b <- c[2,i][[1]]
  if (a == b) { next }
  rv <- getInterLift(yelpdf, "biz_a_state", "biz_b_state", a, b)
  n <- rv$n
  if (n == 0) { next }
  top <- tail(rv$data[order(rv$data$lift),], 1)[c(-1, -11, -12)]
  lift_inter <- rbind(lift_inter, cbind(data.frame(a, b, n), top))
}
write.csv(lift_inter, "network/top_interstate_lift_by_state_pair.csv")

#############
#           #
#  NETWORK  #
#           #
#############

# Get businesses in each state for colors of network graph
PA_biz <- biz$business_id[biz$state == "PA"]
NC_biz <- biz$business_id[biz$state == "NC"]
WI_biz <- biz$business_id[biz$state == "WI"]
IL_biz <- biz$business_id[biz$state == "IL"]
AZ_biz <- biz$business_id[biz$state == "AZ"]
NV_biz <- biz$business_id[biz$state == "NV"]
CA_biz <- biz$business_id[biz$state == "QC" | biz$state == "ON"]

# Make network graph
pairs <- yelpdf[,c("biz_a", "biz_b")]
network <- graph.edgelist(as.matrix(pairs), directed=F)
V(network)$color <- ifelse(
  V(network)$name %in% PA_biz, "red",
  ifelse(V(network)$name %in% NC_biz, "orange",
         ifelse(V(network)$name %in% WI_biz, "yellow",
                ifelse(V(network)$name %in% IL_biz, "green",
                       ifelse(V(network)$name %in% AZ_biz, "blue",
                              ifelse(V(network)$name %in% NV_biz, "purple",
                                     ifelse(V(network)$name %in% CA_biz, "pink",
                                            "gray")))))))
par(mar=c(0,0,0,0))
plot(network, vertex.label=NA, vertex.size=3, edge.curved=F)

# Look at network statistics
d <- degree(network) # Degrees
ddf <- as.data.frame(d)
ddf_top <- labels(ddf)[[1]][ddf$d > median(ddf$d)]
write.csv(ddf_top, "network/top_biz_by_degrees.csv")

b <- betweenness(network) # Betweenness
bdf <- as.data.frame(b)
bdf_top <- labels(bdf)[[1]][bdf$b > median(bdf$b)]
write.csv(bdf_top, "network/top_biz_by_betweenness.csv")

#########################
#                       #
#  AUXILIARY FUNCTIONS  #
#                       #
#########################

################################################################################
# Function to return interstate business pairs with lift                       #
# Args:                                                                        #
#   - df (data.frame): Data frame with business information                    #
#   - col_a (str): Column in df with state information for first business      #
#   - col_b (str): Column in df with state information for second business     #
#   - state_a (str): First state of desired state pair; e.g. "AZ"              #
#   - state_b (str): Second state of desired state pair; e.g. "NV"             #
# Return:                                                                      #
#   - List with keys for data and number of return values                      #
################################################################################

getInterLift <- function(df, col_a, col_b, state_a, state_b) {
  d <- df[df[col_a] == state_a & df[col_b] == state_b,]
  n <- nrow(d)
  rv <- list(data=d, n=n)
  return(rv)
}

################################################################################
# Function to graph neighborhood of network                                    #
# Args:                                                                        #
#   - graph (iGraph graph object): Entity network graph                        #
#   - l (num): Distance of neighborhood; l=2 goes one edge out from center     #
#   - node (str): Name of entity that is the center of the neighborhood        #
#   - manu (bool): If true, indicates that center is a manufacturer; if false, #
#                  center is a physician                                       #
# Return:                                                                      #
#   - Plots neighborhood                                                       #
################################################################################

graphNei <- function(graph, l, node, manu) {
  nei <- graph.neighborhood(graph, l, V(graph)[node])[[1]]
  V(nei)$size <- ifelse(V(nei)$name == node, 6, 3)
  V(nei)$color <- ifelse(V(nei)$name == node, "gold", ifelse(manu, "turquoise", "pink"))
  V(nei)$label.color <- "black"
  V(nei)$label.cex <- ifelse(V(nei)$name == node, 1, .01)
  par(mar=c(0,0,0,0)+.01)
  plot(nei, edge.curved=F)
}

