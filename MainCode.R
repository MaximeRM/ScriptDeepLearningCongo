############################################################################################################
# R Code to predict vegetation types from deep learning approaches (Efficientness B3 or Vision Transformer)
# Code used to generate the map of Picard et al. submitted.
# Written by Maxime Rejou-Mechain the 7 feb 2023

# set working directory
setwd("~/myWorkingDirectorypath")

# Load packages
libs <- c("sf", "terra", "RStoolbox", "svMisc", "caTools", "ggplot2", "dplyr", "tidyr", "keras","tensorflow","reticulate")
lapply(libs,require,character.only=TRUE)

# Check GPU availability
use_backend("tensorflow") 
tf$config$list_physical_devices("GPU")
 
##################################################
## 1- Inputs

# Choose deep learning model to use, should be: "EfficientnetB0", "EfficientnetB3", "EfficientnetB4" or "VITb16"
deepMod <- "EfficientnetB3"

# Create a folder where results will be stored, e.g.: RES_VITb16_7_Feb_2023
outdir <- paste("RES",deepMod, paste(strsplit(date(),split=" ")[[1]][c(3,2,5)],collapse="_"),sep="_")
unlink(outdir,recursive = TRUE)
dir.create(outdir)

# Target size
targSize <- 224

# Path to the PCA image obtained from the script PCAstretch.R
pathToRawImage <- "img255PCAstreched.tiff"

# Path to the cal/val dataset. Should be a shp file (either points or polygons)
pathToPolygones <- "~/myPathToShpfile.shp"

# Path to the project
pathproj <- getwd()

# Name of the variable to be predicted (in the shapefile)
namevarofinterest <- "C_name"

# Classes to be excluded from the cal/val dataset
labexcl <- NULL

# Crop raster of interest to the extent of the calval dataset
cropExt <- FALSE

# Split cal val dataset based on the variable of interest. For instance here 70% cal, 30% val
SplitCalval <- 0.70

# Do not consider images with a X proportion of NA
propNA <- 0.01

# Spatial resol in n pixels
img_width <- 10
img_height <- 10

# Model specification
batch_size <- 32
epochs <- 80

# Restart 
RestartTiling <- FALSE

##################################################
## 2- Load and organize data

# Load the cal/val dataset
poly <- st_read(pathToPolygones)
poly <- poly[!st_is_empty(poly),,drop=F] # remove empty geometries if any
if(!is.null(labexcl)) poly <- poly[as.data.frame(poly)[,namevarofinterest]%in%labexcl,] # remove labels to be excluded
vegclasses <- unique(as.data.frame(poly)[,namevarofinterest]) # Classes to be predicted

# Load raster image
img255 <- rast(pathToRawImage)

# If crop raster
if(cropExt) img255 <- crop(img255,ext(poly))

# Create image folders 
unlink("training",recursive = TRUE)
dir.create("training")
for (i in 1:length(vegclasses)) dir.create(paste0("training/",vegclasses[i]))
unlink("test",recursive = TRUE)
dir.create("test")
for (i in 1:length(vegclasses)) dir.create(paste0("test/",vegclasses[i]))

# split the dataset into 1 training and 1 test dataset and save the calval split
samp <- sample.split(as.data.frame(poly)[,namevarofinterest], SplitRatio=SplitCalval)
save(samp,file=paste0(outdir,'/calvalsplit.rdata'))

# Crop and transform images into RGB png images
print("########### PROCESS TRAINING DATASET")
for(i in 1:length(poly$geometry)){
  imgspat <- vect(poly$geometry[i])
  imgcrop <- crop(img255,imgspat)
  if(sum(is.na(imgcrop[]))/prod(dim(imgcrop))<propNA){
    dircalval <- ifelse(samp[i],"training/","test/")
    writeRaster(imgcrop,paste0(dircalval,as.data.frame(poly)[,namevarofinterest][i],"/","img",i,".png"),overwrite=TRUE)
    progress(i,length(poly$geometry))
  }
}

##################################################
## 3- Deep learning preparation

# number of output vegclasses
output_n <- length(vegclasses)
# image size to scale down to 
target_size <- c(targSize, targSize) # should be 224 for the VIT 16
# RGB = 3 channels
channels <- 3

# path to image folders
train_image_files_path <- file.path(pathproj, "training")
valid_image_files_path <- file.path(pathproj, "test")

# with data augmentation
train_data_gen  <-  image_data_generator(
  rescale = 1/255 ,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  horizontal_flip = TRUE,
  vertical_flip = TRUE,
  fill_mode = "nearest"
)

# Validation data should not be augmented but should be scaled.
valid_data_gen <- image_data_generator(
  rescale = 1/255
)

# training images
train_image_array_gen <- flow_images_from_directory(train_image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = vegclasses)

# validation images
valid_image_array_gen <- flow_images_from_directory(valid_image_files_path, 
                                                    valid_data_gen,                             
                                                    target_size = target_size,
                                                    class_mode = 'categorical',
                                                    classes = vegclasses)

### model definition
# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

################################################################
# Load deep learning model

# Input layer
input <- layer_input(shape = c(target_size, 3))

### 1 load the selected model
if(grepl("Efficientnet",deepMod)){
  reticulate::py_install("efficientnet")
  efn <- import("efficientnet.keras")
  if(deepMod=='EfficientnetB0') conv_base <- efn$EfficientNetB0(weights="noisy-student", include_top=FALSE)
  if(deepMod=='EfficientnetB3') conv_base <- efn$EfficientNetB3(weights="noisy-student", include_top=FALSE)
  if(deepMod=='EfficientnetB4') conv_base <- efn$EfficientNetB4(weights="noisy-student", include_top=FALSE)
  # Define the model
  output <- input %>%
    conv_base %>%
    layer_global_max_pooling_2d() %>%
    layer_batch_normalization() %>%
    layer_dropout(rate=0.5) %>%
    layer_dense(units=output_n, activation="softmax")
}else{
  if(deepMod=='VITb16'){
    # VIT Model saved from the python script vitKeras.py
    modBase=load_model_hdf5('VITMODELb16/')
    # Define the model
    output <- input %>%
      modBase %>%
      layer_batch_normalization() %>% # applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
      layer_dropout(rate=0.5) %>% # randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
      layer_dense(units=output_n, activation="softmax")
  }else{
    stop("Please specify a correct name for the deep learning model")
  }
}

model <- keras_model(input, output)

# Compile the model
model %>% compile(
  optimizer=optimizer_adam(learning_rate = 0.001,
                           use_ema = T,   # Implement exponential moving average
                           ema_momentum = 0.99,
                           ema_overwrite_frequency = NULL),
  loss="categorical_crossentropy",
  metrics = "accuracy"
)

# Initialize a scheduler to reduce the learning rate when the performance on a subset of validation does not improve anymore
lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.1, # Reduction factor
  patience = 10, # Number of epoch before reducing lr
  min_lr = 0.00001 # Minimal learning rate
)

# Fit the model
hist <- model %>% fit(
  # training data
  train_image_array_gen,

  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size),
  epochs = epochs,

  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),

  # print progress
  verbose = 2,
  #
  callbacks = list(lr)
)
# Save the history of the fit
save(hist,file=paste0(outdir,'/histFit.Rdata'))
save_model_hdf5(model, paste0(outdir,'/my_deepmodel.h5'))
     
##################################################
## 4- PREDICT OVER THE FULL IMAGE

# # 1 GENERATE IMAGES
resol <- img_width*img_height
# Generate a template to tile the original image and then to save the predictions
template <- aggregate(subset(img255,1), fact =img_width, na.rm=T)
# Tile the dataset: time consumimg
if(RestartTiling){
  unlink("TileImage",recursive = TRUE)
  dir.create("TileImage")
  dir.create("TileImage/IMG")
  # for (i in 1:length(vegclasses)) dir.create(paste0("TileImage/",vegclasses[i]))
  #
  filename <- paste0("TileImage/IMG/ImgTobePred","_.png")
  tiles <- makeTiles(img255,template, filename, extend=T, datatype="INT1U",overwrite=TRUE)
}

# 2 PREDICT
# Predict over dataset
batch_size <- 2
TobePred_image_files_path <- file.path(pathproj, "TileImage")
pred_datagen <- image_data_generator(rescale = 1/255)
#
test_generator <- flow_images_from_directory(
  TobePred_image_files_path,
  pred_datagen,
  target_size = target_size,
  batch_size = batch_size,
  shuffle = FALSE)
n_samples <- test_generator$n
# Define the classes
classes <- test_generator$classes %>%
  factor() %>%
  table() %>%
  as_tibble()
colnames(classes)[1] <- "value"
# create library of indices & class labels
indices <- train_image_array_gen$class_indices %>%
  as.data.frame() %>%
  gather() %>%
  mutate(value = as.character(value)) %>%
  left_join(classes, by = "value")
# predict on test data
test_generator$reset()
predictions <- model %>%
  predict_generator(
    generator = test_generator,
    steps = as.integer(n_samples/batch_size)
  )
colnames(predictions) <- indices$key
#
IndiceOrder <- as.numeric(stringr::str_match(test_generator$filenames, "Pred_\\s*(.*?)\\s*.png")[,2])
predictions <- predictions[order(IndiceOrder),]
save(predictions,file=paste0(outdir,'/predictions.Rdata'))

# Build and write the predicted map
predictionsVecType <- colnames(predictions)[apply(predictions,1,which.max)]
predictionsVecProb <- apply(predictions,1,max)
rastPredType <- template
rastPredProb <- template
values(rastPredType) <- predictionsVecType
values(rastPredProb) <- predictionsVecProb
rastPredType[is.na(template)] <- NA
rastPredProb[is.na(template)] <- NA
writeRaster(rastPredType,paste0(outdir,'/rastPredType.tiff'))
writeRaster(rastPredProb,paste0(outdir,'/rastPredProb.tiff'))
