############################################################################################################
# R Code to perform a PCA on Sentinel 2 images and then to strech the PCA scores to limit the impact of outliers
# Code used to generate the map of Picard et al. submitted.
# Written by Maxime Rejou-Mechain the 7 feb 2023

# Load the terra library
library(terra)

# Source the strech function
source('strechfun.R')

# Load sentinel image
img <- rast("mySentinelImg.tif")
names(img) <-  c("blue", "green", "red","Veg_red_edge1","Veg_red_edge2","Veg_red_edge3","NIR","Veg_red_edge4","SWIR1","SWIR2")

# Run a PCA on a subset of the image and then predict at the scale of the image
nsampPCA <- 10000
sr  <-  na.omit(spatSample(img,nsampPCA))
PCA  <-  prcomp(sr, scale=TRUE)
rastPCA <- predict(img,PCA)

# Build a stack of stretched PCA axes
rastPCAstrech <- stretch(rastPCA, minq=0.02, maxq=0.98)

# Keep the first 3 PCA axes and generate a 3 band raster
img255 <- subset(rastPCAstrech,1:3)
names(img255) <- c("PCA1", "PCA2", "PCA3")

# Write the raster to be used in the deep learning analyses
writeRaster(img255,"img255PCAstreched.tiff")




