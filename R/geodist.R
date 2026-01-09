#' Calculate euclidean nearest neighbor distances in geographic space or feature space
#'
#' @description Calculates nearest neighbor distances in geographic space or feature space between training data as well as between training data and prediction locations.
#' Optional, the nearest neighbor distances between training data and test data or between training data and CV iterations is computed.
#' @param x object of class sf, training data locations
#' @param modeldomain SpatRaster, stars or sf object defining the prediction area (see Details)
#' @param space "geographical" or "feature". Should the distance be computed in geographic space or in the normalized multivariate predictor space (see Details)
#' @param cvfolds optional. list or vector. Either a list where each element contains the data points used for testing during the cross validation iteration (i.e. held back data).
#' Or a vector that contains the ID of the fold for each training point. See e.g. ?createFolds or ?CreateSpacetimeFolds or ?nndm
#' @param testdata optional. object of class sf: Point data used for independent validation
#' @param preddata optional. object of class sf: Point data indicating the locations within the modeldomain to be used as target prediction points. Useful when the prediction objective is a subset of
#' locations within the modeldomain rather than the whole area.
#' @param samplesize numeric. How many prediction samples should be used?
#' @param sampling character. How to draw prediction samples? See \link[sf]{st_sample} for modeldomains that are sf objects and \link[terra]{spatSample} for raster objects.
#' Use sampling = "Fibonacci" for global applications (only possible with sf objects).
#' @param variables character vector defining the predictor variables used if space="feature". If not provided all variables included in modeldomain are used.
#' @param timevar optional. character. Column that indicates the date. Only used if space="time".
#' @param time_unit optional. Character. Unit for temporal distances See ?difftime.Only used if space="time".
#' @param algorithm see \code{\link[FNN]{knnx.dist}} and \code{\link[FNN]{knnx.index}}
#' @param useMD boolean. Only for `space`=feature: shall the Mahalanobis distance be calculated instead of Euclidean?
#' Only works with numerical variables.
#' @return A data.frame containing the distances. Unit of returned geographic distances is meters. attributes contain W statistic between prediction area and either sample data, CV folds or test data. See details.
#' @details The modeldomain is a sf polygon or a raster that defines the prediction area. The function takes a regular point sample (amount defined by samplesize) from the spatial extent.
#'     If space = "feature", the argument modeldomain (and if provided then also the testdata and/or preddata) has to include predictors. Predictor values for x, testdata and preddata are optional if modeldomain is a raster.
#'     If not provided they are extracted from the modeldomain rasterStack. If some predictors are categorical (i.e., of class factor or character), gower distances will be used.
#'     W statistic describes the match between the distributions. See Linnenbrink et al (2023) for further details.
#' @note See Meyer and Pebesma (2022) for an application of this plotting function
#' @seealso \code{\link{nndm}} \code{\link{knndm}}
#' @import ggplot2
#' @author Hanna Meyer, Edzer Pebesma, Marvin Ludwig, Jan Linnenbrink
#' @examples
#' \dontrun{
#' library(CAST)
#' library(sf)
#' library(terra)
#' library(caret)
#' library(rnaturalearth)
#' library(ggplot2)
#'
#' data(splotdata)
#' studyArea <- rnaturalearth::ne_countries(continent = "South America", returnclass = "sf")
#'
#' ########### Distance between training data and new data:
#' dist <- geodist(splotdata, studyArea)
#' # With density functions
#' plot(dist)
#' # Or ECDFs (relevant for nndm and knnmd methods)
#' plot(dist, stat="ecdf")
#'
#' ########### Distance between training data, new data and test data (here Chile):
#' plot(splotdata[,"Country"])
#' dist <- geodist(splotdata[splotdata$Country != "Chile",], studyArea,
#'                 testdata = splotdata[splotdata$Country == "Chile",])
#' plot(dist)
#'
#' ########### Distance between training data, new data and CV folds:
#' folds <- createFolds(1:nrow(splotdata), k=3, returnTrain=FALSE)
#' dist <- geodist(x=splotdata, modeldomain=studyArea, cvfolds=folds)
#' # Using density functions
#' plot(dist)
#' # Using ECDFs (relevant for nndm and knnmd methods)
#' plot(dist, stat="ecdf")
#'
#' ########### Distances in the feature space:
#' predictors <- terra::rast(system.file("extdata","predictors_chile.tif", package="CAST"))
#' dist <- geodist(x = splotdata,
#'                 modeldomain = predictors,
#'                 space = "feature",
#'                 variables = c("bio_1","bio_12", "elev"))
#' plot(dist)
#'
#' dist <- geodist(x = splotdata[splotdata$Country != "Chile",],
#'                 modeldomain = predictors, cvfolds = folds,
#'                 testdata = splotdata[splotdata$Country == "Chile",],
#'                 space = "feature",
#'                 variables=c("bio_1","bio_12", "elev"))
#' plot(dist)
#'
#'############Distances in temporal space
#' library(lubridate)
#' library(ggplot2)
#' data(cookfarm)
#' dat <- st_as_sf(cookfarm,coords=c("Easting","Northing"))
#' st_crs(dat) <- 26911
#' trainDat <- dat[dat$altitude==-0.3&lubridate::year(dat$Date)==2010,]
#' predictionDat <- dat[dat$altitude==-0.3&lubridate::year(dat$Date)==2011,]
#' trainDat$week <- lubridate::week(trainDat$Date)
#' cvfolds <- CreateSpacetimeFolds(trainDat,timevar = "week")
#'
#' dist <- geodist(trainDat,preddata = predictionDat,cvfolds = cvfolds$indexOut,
#'    space="time",time_unit="days")
#' plot(dist)+ xlim(0,10)
#'
#'
#' ############ Example for a random global dataset
#' ############ (refer to figure in Meyer and Pebesma 2022)
#'
#' ### Define prediction area (here: global):
#' ee <- st_crs("+proj=eqearth")
#' co <- ne_countries(returnclass = "sf")
#' co.ee <- st_transform(co, ee)
#'
#' ### Simulate a spatial random sample
#' ### (alternatively replace pts_random by a real sampling dataset (see Meyer and Pebesma 2022):
#' sf_use_s2(FALSE)
#' pts_random <- st_sample(co.ee, 2000, exact=FALSE)
#'
#' ### See points on the map:
#' ggplot() + geom_sf(data = co.ee, fill="#00BFC4",col="#00BFC4") +
#'   geom_sf(data = pts_random, color = "#F8766D",size=0.5, shape=3) +
#'   guides(fill = "none", col = "none") +
#'   labs(x = NULL, y = NULL)
#'
#' ### plot distances:
#' dist <- geodist(pts_random,co.ee)
#' plot(dist) + scale_x_log10(labels=round)
#'
#'
#'
#'
#'}
#' @export

geodist <- function(x,
                    modeldomain=NULL,
                    space = "geographical",
                    cvfolds=NULL,
                    testdata=NULL,
                    preddata=NULL,
                    samplesize=2000,
                    sampling = "regular",
                    variables=NULL,
                    timevar=NULL,
                    time_unit="auto",
                    algorithm="brute",
                    useMD = FALSE){

  # input formatting ------------
  if(space == "geo") space <- "geographical"
  if(!space %in% c("geographical", "feature")) {
    stop("Space must bei one of 'geographical' or 'feature'")
  }

  if (inherits(modeldomain, "Raster")) {
    modeldomain <- methods::as(modeldomain,"SpatRaster")
  }
  if (inherits(modeldomain, "stars")) {
    if (!requireNamespace("stars", quietly = TRUE))
      stop("package stars required: install that first")
    modeldomain <- methods::as(modeldomain, "SpatRaster")
  }

  # Transform the CRS of the training points to that of the modeldomain if needed
  if (!is.na(sf::st_crs(modeldomain)) && !is.na(sf::st_crs(x)) && sf::st_crs(modeldomain) != sf::st_crs(x)) {
    x <- sf::st_transform(x, sf::st_crs(modeldomain))
  }

  # Extract coordinates of the training locations
  tcoords <- sf::st_coordinates(x)[,1:2]

  if(space == "feature"){

    if(is.null(preddata) && !inherits(modeldomain, "SpatRaster")) {
      stop("Modeldomain must either be a spatRaster object or preddata must be supplied.")
    }
    
    if(is.null(variables)){
      variables <- names(modeldomain)
    }
    
    if(any(!variables%in%names(x))){ # extract variable values of raster:
      message("features are extracted from the modeldomain")
      if(class(x)[1]=="sfc_POINT"){
        x <- sf::st_as_sf(x)
      }
      x <- sf::st_as_sf(terra::extract(modeldomain, terra::vect(x), na.rm=FALSE, bind=TRUE))
    }

    # subset x and modeldomain/preddata to include only the needed predictor values
    x <- x[,variables]

    if(inherits(modeldomain, "SpatRaster")) {
      modeldomain <- modeldomain[[variables]]
    }
  
    if(!is.null(testdata)){
      if(any(!variables%in%names(testdata))){
        # extract variable values of raster:
        testdata <- sf::st_transform(testdata,sf::st_crs(modeldomain))
        testdata <- sf::st_as_sf(terra::extract(modeldomain, terra::vect(testdata), na.rm=FALSE, bind=TRUE))

        if(any(is.na(testdata))){
          testdata <- na.omit(testdata)
          message("some test data were removed because of NA in extracted predictor values")
        }
      }

      testdata <- testdata[,variables]
      if(any(is.na(testdata))){
          testdata <- na.omit(testdata)
          message("some test data were removed because of NA in extracted predictor values")
      }
    }
    if(!is.null(preddata)){
      if(any(!variables%in%names(preddata))){
        # extract variable values of raster:
        preddata <- sf::st_transform(preddata, sf::st_crs(modeldomain))
        preddata <- sf::st_as_sf(terra::extract(modeldomain, terra::vect(preddata), na.rm=FALSE, bind=TRUE))

        if(any(is.na(preddata))){
          preddata <- na.omit(preddata)
          message("some prediction data were removed because of NA in extracted predictor values")
        }
      } 
      modeldomain <- preddata
    }
    # get names of categorical variables
    catVars <- names(x[,variables])[which(sapply(x[,variables], class)%in%c("factor","character"))]
    if(length(catVars)==0) {
      catVars <- NULL
    }
    if(!is.null(catVars)) {
      message(paste0("variable(s) '", catVars, "' is (are) treated as categorical variables"))
    }
  }
  if(space != "feature") {
    catVars <- NULL
  }
  if (space=="time" & is.null(timevar)){
    timevar <- names(which(sapply(x, lubridate::is.Date)))
    message("time variable that has been selected: ",timevar)
  }
  if (space=="time"&time_unit=="auto"){
    time_unit <- units(difftime(sf::st_drop_geometry(x)[,timevar],
                                sf::st_drop_geometry(x)[,timevar]))
  }



  # required steps ----

  ## Sample prediction location from the study area if preddata not available:
  if(is.null(preddata)){
    modeldomain <- sampleFromArea(modeldomain, samplesize, space, variables, sampling, catVars)
  } else{
    modeldomain <- preddata
  }

  # Determine if a projected or geographic CRS was used based on the modeldomain
  if(is.na(sf::st_crs(modeldomain))){
    warning("Missing CRS in training or prediction points. Assuming projected CRS.")
    islonglat <- FALSE
  }else{
    islonglat <- sf::st_is_longlat(modeldomain)
  }

  # Pre-Process data for feature space distance calculation (optional)
  if(space == "feature") {
    x <- sf::st_drop_geometry(x)
    modeldomain <- sf::st_drop_geometry(modeldomain)
    testdata <- sf::st_drop_geometry(testdata)

    if(!is.null(catVars)) {

      # Prepare training data
      x_cat <- x[,catVars,drop=FALSE]
      x_num <- x[,-which(names(x)%in%catVars),drop=FALSE]
      scaleparam <- attributes(scale(x_num))
      x_num <- data.frame(scale(x_num))
      x <- as.data.frame(cbind(x_num, lapply(x_cat, as.factor)))
      x <- x[complete.cases(x),]

      # Prepare modeldomain
      modeldomain_num <- modeldomain[,-which(names(modeldomain)%in%catVars),drop=FALSE]
      modeldomain_cat <- modeldomain[,catVars,drop=FALSE]
      modeldomain_num <- data.frame(scale(modeldomain_num,center=scaleparam$`scaled:center`,
                                          scale=scaleparam$`scaled:scale`))
      modeldomain <- as.data.frame(cbind(modeldomain_num, lapply(modeldomain_cat, as.factor)))

      # Prepare test data
      testdata_num <- testdata[,-which(names(testdata)%in%catVars),drop=FALSE]
      testdata_cat <- testdata[,catVars,drop=FALSE]
      testdata_num <- data.frame(scale(testdata_num,center=scaleparam$`scaled:center`,
                                          scale=scaleparam$`scaled:scale`))
      testdata <- as.data.frame(cbind(testdata_num, lapply(testdata_cat, as.factor)))


    } else {
      scaleparam <- attributes(scale(x))
      x <- data.frame(scale(x))
      x <- x[complete.cases(x),]

      modeldomain <- data.frame(scale(modeldomain,center=scaleparam$`scaled:center`,
                                      scale=scaleparam$`scaled:scale`))
      
      testdata <- data.frame(scale(testdata,center=scaleparam$`scaled:center`,
                                      scale=scaleparam$`scaled:scale`))
    }
  }

  # always do sample-to-sample and sample-to-prediction
  s2s <- sample2sample(x, space,variables,time_unit,timevar, catVars, algorithm=algorithm, useMD = useMD, islonglat=islonglat, tcoords=tcoords)
  s2p <- sample2prediction(x, modeldomain, space, samplesize,variables,time_unit,timevar, catVars, algorithm=algorithm, useMD = useMD, islonglat=islonglat, tcoords=tcoords)

  dists <- rbind(s2s, s2p)

  # optional steps ----
  ##### Distance to test data:
  if(!is.null(testdata)){
    s2t <- sample2test(x, testdata, space,variables,time_unit,timevar, catVars, algorithm=algorithm, useMD = useMD, islonglat=islonglat, tcoords=tcoords)
    dists <- rbind(dists, s2t)
  }

  ##### Distance to CV data:
  if(!is.null(cvfolds)){

    cvd <- cvdistance(x, cvfolds, space, variables,time_unit, timevar, catVars, algorithm=algorithm, useMD = useMD, islonglat=islonglat, tcoords=tcoords)
    dists <- rbind(dists, cvd)
  }
  class(dists) <- c("geodist", class(dists))
  attr(dists, "space") <- space

  if(space=="time"){
    attr(dists, "unit") <- time_unit
  }


  ##### Compute W statistics
  W_sample <- twosamples::wass_stat(dists[dists$what == "sample-to-sample", "dist"],
                                    dists[dists$what == "prediction-to-sample", "dist"])
  attr(dists, "W_sample") <- W_sample
  if(!is.null(testdata)){
    W_test <- twosamples::wass_stat(dists[dists$what == "test-to-sample", "dist"],
                                    dists[dists$what == "prediction-to-sample", "dist"])
    attr(dists, "W_test") <- W_test
  }
  if(!is.null(cvfolds)){
    W_CV <- twosamples::wass_stat(dists[dists$what == "CV-distances", "dist"],
                                  dists[dists$what == "prediction-to-sample", "dist"])
    attr(dists, "W_CV") <- W_CV
  }

  return(dists)
}




# Sample to Sample Distance
sample2sample <- function(x, space, variables, time_unit, timevar, catVars, algorithm, useMD, islonglat, tcoords){
  if(space == "geographical"){

    if(isTRUE(islonglat)){
      distmat <- sf::st_distance(x)
      units(distmat) <- NULL
      diag(distmat) <- NA
      min_d <- apply(distmat, 1, function(x) min(x, na.rm=TRUE))
    }else{
      min_d <- c(FNN::knn.dist(tcoords, k = 1, algorithm=algorithm))
    }

    sampletosample <- data.frame(dist = min_d,
                                 what = factor("sample-to-sample"),
                                 dist_type = "geographical")
  }else if(space == "feature"){
    
    if(is.null(catVars)) {
      if(isTRUE(useMD)) {
        tpoints_mat <- as.matrix(x)

        # use Mahalanobis distances
        if (dim(tpoints_mat)[2] == 1) {
          S <- matrix(stats::var(tpoints_mat), 1, 1)
          tpoints_mat <- as.matrix(tpoints_mat, ncol = 1)
        } else {
          S <- stats::cov(tpoints_mat)
        }
        S_inv <- MASS::ginv(S)

        # calculate distance matrix
        distmat <- matrix(nrow=nrow(x), ncol=nrow(x))
        distmat <- sapply(1:nrow(distmat), function(i) {
          sapply(1:nrow(distmat), function(j) {
            sqrt(t(tpoints_mat[i,] - tpoints_mat[j,]) %*% S_inv %*% (tpoints_mat[i,] - tpoints_mat[j,]))
          })
        })
        diag(distmat) <- NA

        d <- apply(distmat, 1, min, na.rm=TRUE)
      } else {
        d <- c(FNN::knn.dist(x, k = 1, algorithm=algorithm))
      }
    } else {
      # use Gower distances if categorical variables are present
      d <- sapply(1:nrow(x), function(i) gower::gower_topn(x[i,], x[-i,], n=1)$distance[[1]])
    }

    sampletosample <- data.frame(dist = d,
                                 what = factor("sample-to-sample"),
                                 dist_type = "feature")

  }else if(space == "time"){ # calculate temporal distance matrix
    d <- matrix(ncol=nrow(x),nrow=nrow(x))
    for (i in 1:nrow(x)){
      d[i,] <- abs(difftime(sf::st_drop_geometry(x)[,timevar],
                            sf::st_drop_geometry(x)[i,timevar],
                            units=time_unit))
    }
    diag(d) <- Inf
    min_d <- apply(d, 1, min)
    sampletosample <- data.frame(dist = min_d,
                                 what = factor("sample-to-sample"),
                                 dist_type = "time")
  }
  return(sampletosample)
}


# Sample to Prediction
sample2prediction = function(x, modeldomain, space, samplesize, variables, time_unit, timevar, catVars, algorithm, useMD, islonglat, tcoords){

  if(space == "geographical"){

    # ensure that prediction_points and training points have the same CRS
    modeldomain <- sf::st_transform(modeldomain, sf::st_crs(x))

    # calculate the NNDs between prediction points and training points
    if(isTRUE(islonglat)){
      d0 <- sf::st_distance(modeldomain, x)
      units(d0) <- NULL
      min_d0 <- apply(d0, 1, min)
    }else{
      min_d0 <- c(FNN::knnx.dist(query = sf::st_coordinates(modeldomain)[,1:2],
                              data = tcoords, k = 1, algorithm=algorithm))
    }

    sampletoprediction <- data.frame(dist = min_d0,
                                     what = factor("prediction-to-sample"),
                                     dist_type = "geographical")

  }else if(space == "feature"){

    if(is.null(catVars)) {
      
      if(isTRUE(useMD)) {

        tpoints_mat <- as.matrix(x)
        predpoints_mat <- as.matrix(modeldomain)

        # use Mahalanobis distances
        if (dim(tpoints_mat)[2] == 1) {
          S <- matrix(stats::var(tpoints_mat), 1, 1)
          tpoints_mat <- as.matrix(tpoints_mat, ncol = 1)
        } else {
          S <- stats::cov(tpoints_mat)
        }
        S_inv <- MASS::ginv(S)

        target_dist_feature <- sapply(1:dim(predpoints_mat)[1], function(y) {
          min(sapply(1:dim(tpoints_mat)[1], function(x) {
            sqrt(t(predpoints_mat[y,] - tpoints_mat[x,]) %*% S_inv %*% (predpoints_mat[y,] - tpoints_mat[x,]))
          }))
        })
      } else {
        target_dist_feature <- c(FNN::knnx.dist(query = modeldomain, data = x, k = 1, algorithm=algorithm))
      }

    } else {
      target_dist_feature <- c(gower::gower_topn(modeldomain, x, n = 1)$distance)
    }

    sampletoprediction <- data.frame(dist = target_dist_feature,
                                     what = "prediction-to-sample",
                                     dist_type = "feature")
  }else if(space == "time"){

    min_d0 <- c()
    for (i in 1:nrow(modeldomain)){
      min_d0[i] <- min(abs(difftime(sf::st_drop_geometry(modeldomain)[i,timevar],
                                    sf::st_drop_geometry(x)[,timevar],
                                    units=time_unit)))
    }

    sampletoprediction <- data.frame(dist = min_d0,
                                     what = factor("prediction-to-sample"),
                                     dist_type = "time")

  }

  return(sampletoprediction)
}


# sample to test
sample2test <- function(x, testdata, space, variables, time_unit, timevar, catVars, algorithm, useMD, islonglat, tcoords){

  if(space == "geographical"){
    testdata <- sf::st_transform(testdata, sf::st_crs(x))

    # calculate the NNDs between test points and training points
    if(isTRUE(islonglat)){
      d_test <- sf::st_distance(testdata, x)
      units(d_test) <- NULL
      min_d_test <- apply(d_test, 1, min)
    }else{
      min_d_test <- c(FNN::knnx.dist(query = sf::st_coordinates(testdata)[,1:2],
                              data = tcoords, k = 1, algorithm=algorithm))
    }
  
   dists_test <- data.frame(dist = min_d_test,
                             what = factor("test-to-sample"),
                             dist_type = "geographical")


  }else if(space == "feature"){

    if(is.null(catVars)) {
      
      if(isTRUE(useMD)) {

        tpoints_mat <- as.matrix(x)
        testpoints_mat <- as.matrix(testdata)

        # use Mahalanobis distances
        if (dim(tpoints_mat)[2] == 1) {
          S <- matrix(stats::var(tpoints_mat), 1, 1)
          tpoints_mat <- as.matrix(tpoints_mat, ncol = 1)
        } else {
          S <- stats::cov(tpoints_mat)
        }
        S_inv <- MASS::ginv(S)

        test_dist_feature <- sapply(1:dim(testpoints_mat)[1], function(y) {
          min(sapply(1:dim(tpoints_mat)[1], function(x) {
            sqrt(t(testpoints_mat[y,] - tpoints_mat[x,]) %*% S_inv %*% (testpoints_mat[y,] - tpoints_mat[x,]))
          }))
        })
      } else {
        test_dist_feature <- c(FNN::knnx.dist(query = testdata, data = x, k = 1, algorithm=algorithm))
      }

    } else {
      test_dist_feature <- c(gower::gower_topn(testdata, x, n = 1)$distance)
    }

    dists_test <- data.frame(dist = test_dist_feature,
                             what = "test-to-sample",
                             dist_type = "feature")
  }else if (space=="time"){
    min_d0 <- c()
    for (i in 1:nrow(testdata)){
      min_d0[i] <- min(abs(difftime(sf::st_drop_geometry(testdata)[i,timevar],
                                    sf::st_drop_geometry(x)[,timevar],
                                    units=time_unit)))
    }

    dists_test <- data.frame(dist = min_d0,
                             what = factor("test-to-sample"),
                             dist_type = "time")



  }
  return(dists_test)
}



# between folds
cvdistance <- function(x, cvfolds, space, variables, time_unit, timevar, catVars, algorithm, useMD, islonglat, tcoords){

  # Convert cvfold list to vector
  if(!is.null(cvfolds)&is.list(cvfolds)){
    n <- max(unlist(cvfolds))
    clust <- integer(n)

    for (k in seq_along(cvfolds)) {
      clust[cvfolds[[k]]] <- k
    }
  }

  if(space == "geographical"){

    if(isTRUE(islonglat)){
      distmat <- sf::st_distance(x)
      units(distmat) <- NULL
      diag(distmat) <- NA
      d_cv <- distclust_distmat(distmat, clust)
    }else{
      d_cv <- distclust_euclidean(tcoords, clust, algorithm=algorithm)
    }

    dists_cv <- data.frame(dist = d_cv,
                           what = factor("CV-distances"),
                           dist_type = "geographical")


  }else if(space == "feature"){
    x <- sf::st_drop_geometry(x)

    if(!is.null(catVars)) {
      x_cat <- x[,catVars,drop=FALSE]
      x_num <- x[,-which(names(x)%in%catVars),drop=FALSE]
      scaleparam <- attributes(scale(x_num))
      x_num <- data.frame(scale(x_num))
      x <- as.data.frame(cbind(x_num, lapply(x_cat, as.factor)))
      x <- x[complete.cases(x),]
    } else {
      scaleparam <- attributes(scale(x))
      x <- data.frame(scale(x))
      x <- x[complete.cases(x),]
    }

    # Feature space distance calculation between CV folds
    if(is.null(catVars)) {
      if(isTRUE(useMD)) {
        d_cv <- distclust_MD(x, clust)
      } else {
        d_cv <- distclust_euclidean(x, clust, algorithm=algorithm)
      }

    } else {
      d_cv <- distclust_gower(x, clust)
    }

    dists_cv <- data.frame(dist = d_cv,
                           what = factor("CV-distances"),
                           dist_type = "feature")

  }else if(space == "time"){
    d_cv <- c()
    d_cv_tmp <- c()
    for (i in 1:length(cvfolds)){
      for (k in 1:length(cvfolds[[i]])){
        d_cv_tmp[k] <- min(abs(difftime(sf::st_drop_geometry(x)[cvfolds[[i]][k],timevar],
                                        sf::st_drop_geometry(x)[-cvfolds[[i]],timevar],
                                        units=time_unit)))
      }
      
      d_cv <- c(d_cv,d_cv_tmp)
    }


    dists_cv <- data.frame(dist = d_cv,
                           what = factor("CV-distances"),
                           dist_type = "time")

  }

  return(dists_cv)
  }





sampleFromArea <- function(modeldomain, samplesize, space, variables, sampling, catVars){

  # Samples prediction points from the prediction area
  if(inherits(modeldomain, "Raster")){
    modeldomain <- terra::rast(modeldomain)
  }

  if(space == "geographical")  {
    if(inherits(modeldomain, "SpatRaster")) {
        if(samplesize>terra::ncell(modeldomain)){
          samplesize <- terra::ncell(modeldomain)
          message(paste0("samplesize for new data shouldn't be larger than number of pixels.
                  Samplesize was reduced to ",terra::ncell(modeldomain)))
        }
        #create mask to sample from:
        template <- modeldomain[[1]]
        template <- terra::classify(template, cbind(-Inf, Inf, 1), right=FALSE)
        # draw samples using terra
        message(paste0("Sampling ", samplesize, " prediction locations from the modeldomain raster."))
        predictionloc <- terra::spatSample(template, size = samplesize, method = sampling, as.points = TRUE, na.rm = TRUE, values = FALSE) |> 
          sf::st_as_sf()

      }else{
        # sample prediction locations from sf vector objects
        message(paste0("Sampling ", samplesize, " prediction locations from the modeldomain vector."))
        predictionloc <- sf::st_sample(x = modeldomain, size = samplesize, type = sampling) |> 
          sf::st_set_crs(sf::st_crs(modeldomain))
      }
  }

  if(space == "feature"){

    predictionloc <- terra::spatSample(modeldomain, size = samplesize, method = sampling, as.points = TRUE, na.rm = TRUE, values = TRUE) |> 
          sf::st_as_sf()
  }

  return(predictionloc)

}




