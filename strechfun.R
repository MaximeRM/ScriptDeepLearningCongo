############################################################################################################
# R Code adapted from the terra plotRGB function to strech raster values.
# Code used to generate the map of Picard et al. submitted.
# Written by Maxime Rejou-Mechain the 7 feb 2023

stretchRGB=function(x, r=1, g=2, b=3, a=NULL, scale, maxcell=500000, mar=0, stretch=NULL, smooth=FALSE, colNA="white", alpha, bgalpha, addfun=NULL, zlim=NULL, zlimcol=NULL, axes=FALSE, xlab="", ylab="", asp=NULL, add=FALSE, xlim, ylim, ...) {
  x <- x[[c(r, g, b, a)]]
  if (missing(scale)) {
    scale <- 255
    if ( all(hasMinMax(x)) ) {
      rng <- minmax(x)[, 1:3]
      scale <- max(max(rng[2]), 255)
    }
  }
  scale <- as.vector(scale)[1]
  x <- spatSample(x, maxcell, method="regular", as.raster=TRUE)
  if (!is.null(stretch)) {
    if (stretch == "lin") {
      if (!is.null(zlim)) {
        x <- stretch(x, smin=zlim[1], smax=zlim[2])
      } else {
        x <- stretch(x, minq=0.02, maxq=0.98)
      }
    } else {
      x <- stretch(x, histeq=TRUE, scale=255)
    }
    scale <- 255
  }
  RGB <- values(x)
  RGB <- stats::na.omit(RGB)
  naind <- as.vector( attr(RGB, "na.action") )
  if (!is.null(a)) {
    alpha <- RGB[,4] * 255
    RGB <- RGB[,-4]
  }
  if (!is.null(naind)) {
    bg <- grDevices::col2rgb(colNA)
    bg <- grDevices::rgb(bg[1], bg[2], bg[3], alpha=bgalpha, maxColorValue=255)
    z <- rep( bg, times=ncell(x))
    z[-naind] <- grDevices::rgb(RGB[,1], RGB[,2], RGB[,3], alpha=alpha, maxColorValue=scale)
  } else {
    z <- grDevices::rgb(RGB[,1], RGB[,2], RGB[,3], alpha=alpha, maxColorValue=scale)
  }
  z <- matrix(z, nrow=nrow(x), ncol=ncol(x), byrow=TRUE)
  requireNamespace("grDevices")
  bb <- as.vector(matrix(as.vector(ext(x)), ncol=2))
  bb <- as.vector(ext(x))
  if (!add) {
    if (is.null(asp)) {
      if (is.lonlat(x, perhaps=TRUE, warn=FALSE)) {
        ym <- mean(bb[3:4])
        asp <- 1/cos((ym * pi)/180)
      } else {
        asp <- 1
      }
    }
    if (missing(xlim)) xlim=c(bb[1], bb[2])
    if (missing(ylim)) ylim=c(bb[3], bb[4])
    plot(NA, NA, xlim=xlim, ylim=ylim, type = "n", xaxs='i', yaxs='i', xlab=xlab, ylab=ylab, asp=asp, axes=FALSE, ...)
    if (axes) {
      xticks <- graphics::axTicks(1, c(xlim[1], xlim[2], 4))
      yticks <- graphics::axTicks(2, c(ylim[1], ylim[2], 4))
      if (xres(x) %% 1 == 0) xticks = round(xticks)
      if (yres(x) %% 1 == 0) yticks = round(yticks)
      graphics::axis(1, at=xticks)
      graphics::axis(2, at=yticks, las = 1)
      graphics::box()
    }
  }
  graphics::rasterImage(z, bb[1], bb[3], bb[2], bb[4], interpolate=smooth, ...)
  if (!is.null(addfun)) {
    if (is.function(addfun)) {
      addfun()
    }
  }
}


..linStretch <- function (x) {
  v <- stats::quantile(x, c(0.02, 0.98), na.rm = TRUE)
  temp <- (255 * (x - v[1]))/(v[2] - v[1])
  temp[temp < 0] <- 0
  temp[temp > 255] <- 255
  return(temp)
}

# Histogram equalization stretch
..eqStretch <- function(x){
  ecdfun <- stats::ecdf(x)
  ecdfun(x)*255
}

..rgbstretch <- function(RGB, stretch, caller="") {
  stretch = tolower(stretch)
  if (stretch == 'lin') {
    RGB[,1] <- ..linStretch(RGB[,1])
    RGB[,2] <- ..linStretch(RGB[,2])
    RGB[,3] <- ..linStretch(RGB[,3])
  } else if (stretch == 'hist') {
    RGB[,1] <- ..eqStretch(RGB[,1])
    RGB[,2] <- ..eqStretch(RGB[,2])
    RGB[,3] <- ..eqStretch(RGB[,3])
  } else if (stretch != '') {
    warn(caller, "invalid stretch value")
  }
  RGB
}
