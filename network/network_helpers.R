###############################################################################
#                                                                             #
#  CMSC 12300                                                                 #
#  Project: Yelp Dataset Challenge                                            #
#  Helpers to network.R                                                       #
#  Coded by Scarlett Swerdlow                                                 #
#  scarlettswerdlow@uchicago.edu                                              #
#  June 3, 2015                                                               #
#                                                                             #
###############################################################################

###############
#             #
#  CONSTANTS  #
#             #
###############

EDGES_FN <- "network/yelp_network.csv"
BIZ_FN <- "yelp_academic_dataset_business.json"

##################
#                #
#  DEPENDENCIES  #
#                #
##################

# Use data.table package to quickly read in csv
if (require("data.table")) {
  cat("data.table is loaded correctly \n")
} else {
  cat("Trying to install data.table \n")
  install.packages("data.table")
  if (require("data.table")) {
    cat("data.table successfully installed and loaded \n")
  } else {
    stop("Cound not install data.table")
  }
}

# Use igraph package for network analysis
if (require("igraph")) {
  cat("igraph is loaded correctly \n")
} else {
  cat("Trying to install igraph \n")
  install.packages("igraph")
  if (require("igraph")) {
    cat("igraph successfully installed and loaded \n")
  } else {
    stop("Cound not install igraph")
  }
}

# Use jsonlite package to quickly read in csv
if (require("jsonlite")) {
  cat("jsonlite is loaded correctly \n")
} else {
  cat("Trying to install jsonlite \n")
  install.packages("jsonlite")
  if (require("jsonlite")) {
    cat("jsonlite successfully installed and loaded \n")
  } else {
    stop("Cound not install jsonlite")
  }
}

###############
#             #
#  FUNCTIONS  #
#             #
###############

loadAndCleanData <- function(edges_fn, biz_fn) {
  yelp_rv <- fread(edges_fn, header=T)
  yelp_rv <- as.data.frame(yelp_rv)
  names(yelp_rv) <- c("row", "biz_a", "biz_b", "prob_a", "prob_b", "prob_ab", 
                 "conf", "lift")
  
  # Add some business data
  biz_rv <- stream_in(file(biz_fn))
  yelp_rv$biz_a_name <- biz_rv$name[match(yelp_rv$biz_a, biz_rv$business_id)]
  yelp_rv$biz_b_name <- biz_rv$name[match(yelp_rv$biz_b, biz_rv$business_id)]
  yelp_rv$biz_a_state <- biz_rv$state[match(yelp_rv$biz_a, biz_rv$business_id)]
  yelp_rv$biz_b_state <- biz_rv$state[match(yelp_rv$biz_b, biz_rv$business_id)]
  
  # Prepare vectors of businesses by state for network analysis
  PA_biz <- biz_rv$business_id[biz_rv$state == "PA"]
  NC_biz <- biz_rv$business_id[biz_rv$state == "NC"]
  WI_biz <- biz_rv$business_id[biz_rv$state == "WI"]
  IL_biz <- biz_rv$business_id[biz_rv$state == "IL"]
  AZ_biz <- biz_rv$business_id[biz_rv$state == "AZ"]
  NV_biz <- biz_rv$business_id[biz_rv$state == "NV"]
  CA_biz <- biz_rv$business_id[biz_rv$state == "QC" | biz_rv$state == "ON"]
  
  rv <- list(yelp=yelp_rv, biz=biz_rv, pa=PA_biz, nc=NC_biz, wi=WI_biz,
             il=IL_biz, az=AZ_biz, nv=NV_biz, ca=CA_biz)
  
  return(rv)
}

getTopLift <- function(edges_df, k, output=F, output_fn) {
  rv <- tail(edges_df[order(edges_df$lift),], k)[,-1]
  if (output) { write.csv(rv, output_fn) }
  return(rv)
}

getTopLiftState <- function(edges_df) {
  state <- edges_df$biz_a_state
  x <- edges_df[edges_df$biz_b_state == state,]
  if (nrow(x) == 0) { return(NA) }
  rv <- x[which.max(x$lift),]
  return(rv)
}

getTopLiftStateHelper <- function(edges_df, by_col, output=F, output_fn) {
  rv <- by(edges_df, edges_df[by_col], getTopLiftState)
  rv <- do.call("rbind", rv) # Change to dataframe
  if (output) { write.csv(rv, output_fn) }
  return(rv)
}

getTopLiftInter <- function(edges_df, col_a, col_b, state_a, state_b) {
  d <- edges_df[edges_df[col_a] == state_a & edges_df[col_b] == state_b,]
  n <- nrow(d)
  rv <- list(data=d, n=n)
  return(rv)
}

getTopLiftInterHelper <- function(edges_df, output=F, output_fn) {
  rv <- data.frame()
  c <- combn(unique(c(edges_df$biz_a_state, edges_df$biz_b_state)),2)
  for (i in 1:ncol(c)) {
    a <- c[1,i][[1]]
    b <- c[2,i][[1]]
    if (a == b) { next }
    x <- getTopLiftInter(edges_df, "biz_a_state", "biz_b_state", a, b)
    n <- x$n
    if (n == 0) { next }
    top <- tail(x$data[order(x$data$lift),], 1)[c(-1, -11, -12)]
    rv <- rbind(rv, cbind(data.frame(a, b, n), top))
  }
  if (output) { write.csv(rv, output_fn) }
  return(rv)
}

plotGraph <- function(edges_df, plot=F){
  # Assume data list from loadAndCleanData is in environment
  pairs <- edges_df[,c("biz_a", "biz_b")]
  network <- graph.edgelist(as.matrix(pairs), directed=F)
  V(network)$color <- ifelse(
    V(network)$name %in% data$pa, "red",
    ifelse(V(network)$name %in% data$nc, "orange",
           ifelse(V(network)$name %in% data$wi, "yellow",
                  ifelse(V(network)$name %in% data$il, "green",
                         ifelse(V(network)$name %in% data$az, "blue",
                                ifelse(V(network)$name %in% data$nv, "purple",
                                       ifelse(V(network)$name %in% data$ca, "pink",
                                              "gray")))))))
  if (plot) { plot(network, vertex.label=NA, vertex.size=3, edge.curved=F) }
  return(network)
}

plotNeighborhood <- function(network, l, node, manu) {
  nei <- graph.neighborhood(network, l, V(network)[node])[[1]]
  V(nei)$size <- ifelse(V(nei)$name == node, 6, 3)
  V(nei)$color <- ifelse(V(nei)$name == node, "gold", ifelse(manu, "turquoise", "pink"))
  V(nei)$label.color <- "black"
  V(nei)$label.cex <- ifelse(V(nei)$name == node, 1, .01)
  par(mar=c(0,0,0,0)+.01)
  plot(nei, edge.curved=F)
}

getDegrees <- function(network, output=F, output_fn) {
  d <- degree(network)
  ddf <- as.data.frame(d)
  rv <- labels(ddf)[[1]][ddf$d > median(ddf$d)]
  if (output) { write.csv(rv, output_fn) }
  return(rv)
}

getBetweenness <- function(network, output=F, output_fn) {
  b <- betweenness(network)
  bdf <- as.data.frame(b)
  rv <- labels(bdf)[[1]][bdf$b > median(bdf$b)]
  if (output) { write.csv(rv, output_fn) }
  return(rv)
}



