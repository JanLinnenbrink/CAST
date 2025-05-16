#' K-fold Nearest Neighbour Distance Matching
#' @description
#' This function implements the kNNDM algorithm and returns the necessary
#' indices to perform a k-fold NNDM CV for map validation.
#'
#' @author Carles Milà and Jan Linnenbrink
#' @param tpoints sf or sfc point object, or data.frame if space = "feature". Contains the training points samples.
#' @param modeldomain sf polygon object or SpatRaster defining the prediction area. Optional; alternative to predpoints (see Details).
#' @param predpoints sf or sfc point object, or data.frame if space = "feature". Contains the target prediction points. Optional; alternative to modeldomain (see Details).
#' @param space character. Either "geographical" or "feature".
#' @param k integer. Number of folds desired for CV. Defaults to 10.
#' @param prop_test numeric. The proportion of data that should be assigned to the test set (train/test split).
#' By default NULL, resulting in k-fold Cross-Validation instead of train/test split.
#' @param maxp numeric. Maximum fold size allowed, defaults to 0.5, i.e. a single fold can hold a maximum of half of the training points.
#' @param clustering character. Possible values include "hierarchical" and "kmeans". See details.
#' @param linkf character. Only relevant if clustering = "hierarchical". Link function for agglomerative hierarchical clustering.
#' Defaults to "ward.D2". Check `stats::hclust` for other options.
#' @param samplesize numeric. How many points in the modeldomain should be sampled as prediction points?
#' Only required if modeldomain is used instead of predpoints.
#' @param sampling character. How to draw prediction points from the modeldomain? See `sf::st_sample`.
#' Only required if modeldomain is used instead of predpoints.
#' @param useMD boolean. Only for `space`=feature: shall the Mahalanobis distance be calculated instead of Euclidean?
#' Only works with numerical variables.
#' @return An object of class \emph{knndm} consisting of a list of eight elements:
#' indx_train, indx_test (indices of the observations to use as
#' training/test data in each kNNDM CV iteration), Gij (distances for
#' G function construction between prediction and target points), Gj
#' (distances for G function construction during LOO CV), Gjstar (distances
#' for modified G function during kNNDM CV), clusters (list of cluster IDs),
#' W (Wasserstein statistic), and space (stated by the user in the function call).
#'
#' @details
#' knndm is a k-fold version of NNDM LOO CV for medium and large datasets. Brielfy, the algorithm tries to
#' find a k-fold configuration such that the integral of the absolute differences (Wasserstein W statistic)
#' between the empirical nearest neighbour distance distribution function between the test and training data during CV (Gj*),
#' and the empirical nearest neighbour distance distribution function between the prediction and training points (Gij),
#' is minimised. It does so by performing clustering of the training points' coordinates for different numbers of
#' clusters that range from k to N (number of observations), merging them into k final folds,
#' and selecting the configuration with the lowest W.
#'
#' Using a projected CRS in `knndm` has large computational advantages since fast nearest neighbour search can be
#' done via the `FNN` package, while working with geographic coordinates requires computing the full
#' spherical distance matrices. As a clustering algorithm, `kmeans` can only be used for
#' projected CRS while `hierarchical` can work with both projected and geographical coordinates, though it requires
#' calculating the full distance matrix of the training points even for a projected CRS.
#'
#' In order to select between clustering algorithms and number of folds `k`, different `knndm` configurations can be run
#' and compared, being the one with a lower W statistic the one that offers a better match. W statistics between `knndm`
#' runs are comparable as long as `tpoints` and `predpoints` or `modeldomain` stay the same.
#'
#' Map validation using `knndm` should be used using `CAST::global_validation`, i.e. by stacking all out-of-sample
#' predictions and evaluating them all at once. The reasons behind this are 1) The resulting folds can be
#' unbalanced and 2) nearest neighbour functions are constructed and matched using all CV folds simultaneously.
#'
#' If training data points are very clustered with respect to the prediction area and the presented `knndm`
#' configuration still show signs of Gj* > Gij, there are several things that can be tried. First, increase
#' the `maxp` parameter; this may help to control for strong clustering (at the cost of having unbalanced folds).
#' Secondly, decrease the number of final folds `k`, which may help to have larger clusters.
#'
#' The `modeldomain` is either a sf polygon that defines the prediction area, or alternatively a SpatRaster out of which a polygon,
#' transformed into the CRS of the training points, is defined as the outline of all non-NA cells.
#' Then, the function takes a regular point sample (amount defined by `samplesize`) from the spatial extent.
#' As an alternative use `predpoints` instead of `modeldomain`, if you have already defined the prediction locations (e.g. raster pixel centroids).
#' When using either `modeldomain` or `predpoints`, we advise to plot the study area polygon and the training/prediction points as a previous step to ensure they are aligned.
#'
#' `knndm` can also be performed in the feature space by setting `space` to "feature".
#' Euclidean distances or Mahalanobis distances can be used for distance calculation, but only Euclidean are tested.
#' In this case, nearest neighbour distances are calculated in n-dimensional feature space rather than in geographical space.
#' `tpoints` and `predpoints` can be data frames or sf objects containing the values of the features. Note that the names of `tpoints` and `predpoints` must be the same.
#' `predpoints` can also be missing, if `modeldomain` is of class SpatRaster. In this case, the values of of the SpatRaster will be extracted to the `predpoints`.
#' In the case of any categorical features, Gower distances will be used to calculate the Nearest Neighbour distances [Experimental]. If categorical
#' features are present, and `clustering` = "kmeans", K-Prototype clustering will be performed instead.
#'
#' @references
#' \itemize{
#' \item Linnenbrink, J., Milà, C., Ludwig, M., and Meyer, H.: kNNDM: k-fold Nearest Neighbour Distance Matching Cross-Validation for map accuracy estimation, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2023-1308, 2023.
#' \item Milà, C., Mateu, J., Pebesma, E., Meyer, H. (2022): Nearest Neighbour Distance Matching Leave-One-Out Cross-Validation for map validation. Methods in Ecology and Evolution 00, 1– 13.
#' }
#' @seealso \code{\link{geodist}}, \code{\link{nndm}}
#'
#' @export
#' @examples
#' ########################################################################
#' # Example 1: Simulated data - Randomly-distributed training points
#' ########################################################################
#'
#' library(sf)
#' library(ggplot2)
#'
#' # Simulate 1000 random training points in a 100x100 square
#' set.seed(1234)
#' simarea <- list(matrix(c(0,0,0,100,100,100,100,0,0,0), ncol=2, byrow=TRUE))
#' simarea <- sf::st_polygon(simarea)
#' train_points <- sf::st_sample(simarea, 1000, type = "random")
#' pred_points <- sf::st_sample(simarea, 1000, type = "regular")
#' plot(simarea)
#' plot(pred_points, add = TRUE, col = "blue")
#' plot(train_points, add = TRUE, col = "red")
#'
#' # Run kNNDM for the whole domain, here the prediction points are known.
#' # Firstly split in train/test
#' knndm_test <- knndm(train_points, predpoints = pred_points, k = 2, prop_test=0.3)
#' # Then, split train data into CV folds
#' train_points_split <- train_points[knndm_test$clusters == 1,]
#' knndm_folds <- knndm(train_points_split, predpoints = pred_points, k = 10)
#' knndm_folds
#' plot(knndm_folds)
#' plot(knndm_folds, type = "simple") # For more accessible legend labels
#' plot(knndm_folds, type = "simple", stat = "density") # To visualize densities rather than ECDFs
#' folds <- as.character(knndm_folds$clusters)
#' ggplot() +
#'   geom_sf(data = simarea, alpha = 0) +
#'   geom_sf(data = train_points_split, aes(col = folds)) +
#'   geom_sf(data = train_points[knndm_test$cluster == 2,], colour="black")
#'
#' ########################################################################
#' # Example 2: Simulated data - Clustered training points
#' ########################################################################
#' \dontrun{
#' library(sf)
#' library(ggplot2)
#'
#' # Simulate 1000 clustered training points in a 100x100 square
#' set.seed(1234)
#' simarea <- list(matrix(c(0,0,0,100,100,100,100,0,0,0), ncol=2, byrow=TRUE))
#' simarea <- sf::st_polygon(simarea)
#' train_points <- clustered_sample(simarea, 1000, 50, 5)
#' pred_points <- sf::st_sample(simarea, 1000, type = "regular")
#' plot(simarea)
#' plot(pred_points, add = TRUE, col = "blue")
#' plot(train_points, add = TRUE, col = "red")
#'
#' # Run kNNDM for the whole domain, here the prediction points are known.
#' # Firstly split in train/test
#' knndm_test <- knndm(train_points, predpoints = pred_points, k = 2, prop_test=0.3)
#' # Then, split train data into CV folds
#' train_points_split <- train_points[knndm_test$clusters == 1,]
#' knndm_folds <- knndm(train_points_split, predpoints = pred_points, k = 10)
#' knndm_folds
#' plot(knndm_test)
#' plot(knndm_folds)
#' plot(knndm_folds, type = "simple") # For more accessible legend labels
#' plot(knndm_folds, type = "simple", stat = "density") # To visualize densities rather than ECDFs
#' folds <- as.character(knndm_folds$clusters)
#' ggplot() +
#'   geom_sf(data = simarea, alpha = 0) +
#'   geom_sf(data = train_points_split, aes(col = folds)) +
#'   geom_sf(data = train_points[knndm_test$cluster == 2,], colour="black")
#'}
#' ########################################################################
#' # Example 3: Real- world example; using a modeldomain instead of previously
#' # sampled prediction locations
#' ########################################################################
#' \dontrun{
#' library(sf)
#' library(terra)
#' library(ggplot2)
#'
#' ### prepare sample data:
#' data(cookfarm)
#' dat <- aggregate(cookfarm[,c("DEM","TWI", "NDRE.M", "Easting", "Northing","VW")],
#'    by=list(as.character(cookfarm$SOURCEID)),mean)
#' pts <- dat[,-1]
#' pts <- st_as_sf(pts,coords=c("Easting","Northing"))
#' st_crs(pts) <- 26911
#' studyArea <- rast(system.file("extdata","predictors_2012-03-25.tif",package="CAST"))
#' pts <- st_transform(pts, crs = st_crs(studyArea))
#' terra::plot(studyArea[["DEM"]])
#' terra::plot(vect(pts), add = T)
#'
#' knndm_folds <- knndm(pts, modeldomain=studyArea, k = 5)
#' knndm_folds
#' plot(knndm_folds)
#' folds <- as.character(knndm_folds$clusters)
#' ggplot() +
#'   geom_sf(data = pts, aes(col = folds))
#'
#' #use for cross-validation:
#' library(caret)
#' ctrl <- trainControl(method="cv",
#'    index=knndm_folds$indx_train,
#'    savePredictions='final')
#' model_knndm <- train(dat[,c("DEM","TWI", "NDRE.M")],
#'    dat$VW,
#'    method="rf",
#'    trControl = ctrl)
#' global_validation(model_knndm)
#'}
#' ########################################################################
#' # Example 4: Real- world example; kNNDM in feature space
#' ########################################################################
#' \dontrun{
#' library(sf)
#' library(terra)
#' library(ggplot2)
#'
#'data(splotdata)
#'splotdata <- splotdata[splotdata$Country == "Chile",]
#'
#'predictors <- c("bio_1", "bio_4", "bio_5", "bio_6",
#'                "bio_8", "bio_9", "bio_12", "bio_13",
#'                "bio_14", "bio_15", "elev")
#'
#'trainDat <- sf::st_drop_geometry(splotdata)
#'predictors_sp <- terra::rast(system.file("extdata", "predictors_chile.tif",package="CAST"))
#'
#'
#' terra::plot(predictors_sp[["bio_1"]])
#' terra::plot(vect(splotdata), add = T)
#'
#'knndm_folds <- knndm(trainDat[,predictors], modeldomain = predictors_sp, space = "feature",
#'                     clustering="kmeans", k=4, maxp=0.8)
#'plot(knndm_folds, type="simple")
#'
#'#' ########################################################################
#' # Example 5: Real- world example; kNNDM in feature and geographical space
#' ########################################################################
#' \dontrun{
#' library(sf)
#' library(terra)
#' library(ggplot2)
#'
#'data(splotdata)
#'splotdata <- splotdata[splotdata$Country == "Chile",]
#'
#'predictors <- c("bio_1", "bio_4", "bio_5", "bio_6",
#'                "bio_8", "bio_9", "bio_12", "bio_13",
#'                "bio_14", "bio_15", "elev")
#'
#'trainDat <- splotdata
#'predictors_sp <- terra::rast(system.file("extdata", "predictors_chile.tif",package="CAST"))
#'
#'
#' terra::plot(predictors_sp[["bio_1"]])
#' terra::plot(vect(splotdata), add = T)
#'
#'knndm_folds <- knndm(trainDat[,predictors], modeldomain = predictors_sp, space = "feature_geo",
#'                     clustering="hierarchical", k=4, maxp=0.8)
#'plot(knndm_folds, type="simple")
#'
#'}
knndm <- function(tpoints, modeldomain = NULL, predpoints = NULL,
                  space = "geographical",
                  k = 10, maxp = 0.5, prop_test=NULL,
                  clustering = "hierarchical", linkf = "ward.D2",
                  samplesize = 1000, sampling = "regular", useMD=FALSE,
                  lambda = 1){


  # create sample points from modeldomain
  if(is.null(predpoints)&!is.null(modeldomain)){

    # Check modeldomain is indeed a sf/SpatRaster
    if(!any(c("sfc", "sf", "SpatRaster") %in% class(modeldomain))){
      stop("modeldomain must be a sf/sfc object or a 'SpatRaster' object.")
    }

    # If modeldomain is a SpatRaster, transform into polygon
    if(any(class(modeldomain) == "SpatRaster")){

      # save predictor stack for extraction if space = "feature"
      if(space == "feature" | space=="feature_geo") {
        predictor_stack <- modeldomain
      }
      modeldomain[!is.na(modeldomain)] <- 1
      modeldomain <- terra::as.polygons(modeldomain, values = FALSE, na.all = TRUE) |>
        sf::st_as_sf() |>
        sf::st_union()
      if(any(c("sfc", "sf") %in% class(tpoints))) {
        modeldomain <- sf::st_transform(modeldomain, crs = sf::st_crs(tpoints))
      }
    }




    # Check modeldomain is indeed a polygon sf
    if(!any(class(sf::st_geometry(modeldomain)) %in% c("sfc_POLYGON", "sfc_MULTIPOLYGON"))){
      stop("modeldomain must be a sf/sfc polygon object.")
    }

    # Check whether modeldomain has the same crs as tpoints
    if(!identical(sf::st_crs(tpoints), sf::st_crs(modeldomain)) & space == "geographical"){
      stop("tpoints and modeldomain must have the same CRS")
    }

    # We sample
    message(paste0(samplesize, " prediction points are sampled from the modeldomain"))
    predpoints <- sf::st_sample(x = modeldomain, size = samplesize, type = sampling)
    sf::st_crs(predpoints) <- sf::st_crs(modeldomain)

    if(space == "feature") {
      message("predictor values are extracted for prediction points")
      predpoints <- terra::extract(predictor_stack, terra::vect(predpoints), ID=FALSE)
    } else if(space == "feature_geo") {
      message("predictor values are extracted for prediction points")
      predpoints <- terra::extract(predictor_stack, terra::vect(predpoints), ID=FALSE, bind=TRUE) |>
        sf::st_as_sf()
    }

  }else if(!is.null(predpoints) & space %in% c("geographical", "feature_geo")){
    if(!identical(sf::st_crs(tpoints), sf::st_crs(predpoints))){
      stop("tpoints and predpoints must have the same CRS")
    }
  }


  # Conditional preprocessing actions
  if(space %in% c("geographical", "feature_geo")) {
    if (any(class(tpoints) %in% "sfc")) {
      tpoints <- sf::st_sf(geom = tpoints)
    }
    if (any(class(predpoints) %in% "sfc")) {
      predpoints <- sf::st_sf(geom = predpoints)
    }
    if(is.na(sf::st_crs(tpoints))){
      warning("Missing CRS in training or prediction points. Assuming projected CRS.")
      islonglat <- FALSE
    }else{
      islonglat <- sf::st_is_longlat(tpoints)
    }
  }
  if(space == "feature") {
    # drop geometry if tpoints / predpoints are of class sf
    if(any(class(tpoints) %in% c("sf","sfc")) & space == "feature") {
      tpoints <- sf::st_set_geometry(tpoints, NULL)
    }
    if(any(class(predpoints) %in% c("sf","sfc")) & space == "feature") {
      predpoints <- sf::st_set_geometry(predpoints, NULL)
    }
  }
  if (space %in% c("feature", "feature_geo")) {
    # get names of categorical variables
    catVars <- names(tpoints)[which(sapply(tpoints, class)%in%c("factor","character"))]
    if(length(catVars)==0) {
      catVars <- NULL
    }
    if(!is.null(catVars)) {
      message(paste0("variable(s) '", catVars, "' is (are) treated as categorical variables"))
    }
    # omit NAs
    if(any(is.na(predpoints))) {
      message("some prediction points contain NAs, which will be removed")
      predpoints <- stats::na.omit(predpoints)
    }
    if(any(is.na(tpoints))) {
      message("some training points contain NAs, which will be removed")
      tpoints <- stats::na.omit(tpoints)
    }
  }


  # if train/test split desired: adjust parameters
  if(!is.null(prop_test)) {
    k <- 2
    if(maxp <= 0.5) {
      maxp <- 0.9
    }
  }


  # kNNDM in the geographical / feature space
  if(isTRUE(space == "geographical")){

    # prior checks
    check_knndm_geo(tpoints, predpoints, space, k, maxp, clustering, islonglat)
    # kNNDM in geographical space
    knndm_res <- knndm_geo(tpoints, predpoints, k, maxp, clustering, linkf, islonglat, prop_test)

  } else if (isTRUE(space == "feature")) {

    # prior checks
    check_knndm_feature(tpoints, predpoints, space, k, maxp, clustering, islonglat, catVars, useMD)
    # kNNDM in feature space
    knndm_res <- knndm_feature(tpoints, predpoints, k, maxp, clustering, linkf, catVars, useMD, prop_test)

  } else if (isTRUE(space == "feature_geo")) {

    # prior checks
    check_knndm_feature(tpoints, predpoints, space, k, maxp, clustering, islonglat, catVars, useMD)
    check_knndm_geo(tpoints, predpoints, space, k, maxp, clustering, islonglat)

    # kNNDM in feature space
    knndm_res <- knndm_feature_geo(tpoints, predpoints, k, maxp, clustering, linkf, catVars, useMD, prop_test,
                                   islonglat, lambda)

  }

  # Output
  knndm_res
}


# kNNDM checks
check_knndm_geo <- function(tpoints, predpoints, space, k, maxp, clustering, islonglat){

  if(!identical(sf::st_crs(tpoints), sf::st_crs(predpoints))){
    warning("tpoints and predpoints must have the same CRS; predpoints have been transformed to the CRS of tpoints")
    predpoints <- sf::st_transform(predpoints, sf::st_crs(tpoints))
  }
  if (!(clustering %in% c("kmeans", "hierarchical"))) {
    stop("clustering must be one of `kmeans` or `hierarchical`")
  }
  if (!(maxp < 1 & maxp > 1/k)) {
    stop("maxp must be strictly between 1/k and 1")
  }
  if(isTRUE(islonglat) & clustering == "kmeans"){
    stop("kmeans works in the Euclidean space and therefore can only handle
         projected coordinates. Please use hierarchical clustering or project your data.")
  }
}

check_knndm_feature <- function(tpoints, predpoints, space, k, maxp, clustering, islonglat, catVars, useMD){

  if(!is.null(catVars) & isTRUE(useMD)) {
    warning("Mahalanobis distances not supported for categorical features, Gower distances will be used")
    useMD <- FALSE
  }

  if (!(maxp < 1 & maxp > 1/k)) {
    stop("maxp must be strictly between 1/k and 1")
  }

  if(is.null(predpoints)) {
    stop("predpoints with predictor data missing")
  }

  if(length(setdiff(names(tpoints), names(predpoints)))>0) {
    stop("tpoints and predpoints need to contain the predictor data and have the same colnames.")
  }

  for (catvar in catVars) {
    if (any(!unique(tpoints[,catvar]) %in% unique(predpoints[,catvar]))) {
      stop(paste0("Some values of factor", catvar, "are only present in training / prediction points.
                  All factor values in the prediction points must be present in the training points."))
    }
  }

}


# kNNDM in the geographical space
knndm_geo <- function(tpoints, predpoints, k, maxp, clustering, linkf, islonglat, prop_test){

  # Gj and Gij calculation
  tcoords <- sf::st_coordinates(tpoints)[,1:2]
  if(isTRUE(islonglat)){
    distmat <- sf::st_distance(tpoints)
    units(distmat) <- NULL
    diag(distmat) <- NA
    Gj <- apply(distmat, 1, function(x) min(x, na.rm=TRUE))
    Gij <- sf::st_distance(predpoints, tpoints)
    units(Gij) <- NULL
    Gij <- apply(Gij, 1, min)
  }else{
    Gj <- c(FNN::knn.dist(tcoords, k = 1))
    Gij <- c(FNN::knnx.dist(query = sf::st_coordinates(predpoints)[,1:2],
                            data = tcoords, k = 1))
  }

  # Check if Gj > Gij (warning suppressed regarding ties)
  testks <- suppressWarnings(stats::ks.test(Gj, Gij, alternative = "great"))
  if(testks$p.value >= 0.05){


    if(!is.null(prop_test)) {
      clust <- sample(rep(1:k, ceiling(nrow(tpoints)/k)), size = nrow(tpoints), replace=T,
                      prob = rep(c(1-prop_test, prop_test), ceiling(nrow(tpoints)/k)))
    } else {
      clust <- sample(rep(1:k, ceiling(nrow(tpoints)/k)), size = nrow(tpoints), replace=F)
    }

    if(isTRUE(islonglat)){
      Gjstar <- distclust_distmat(distmat, clust)
    }else{
      Gjstar <- distclust_euclidean(tcoords, clust)
    }
    k_final <- "random CV"
    W_final <- twosamples::wass_stat(Gjstar, Gij)
    message("Gij <= Gj; a random CV assignment is returned")

  }else{

    if(clustering == "hierarchical"){
      # For hierarchical clustering we need to compute the full distance matrix,
      # but we can integrate geographical distances
      if(!isTRUE(islonglat)){
        distmat <- sf::st_distance(tpoints)
      }
      hc <- stats::hclust(d = stats::as.dist(distmat), method = linkf)
    }

    # Build grid of number of clusters to try - we sample low numbers more intensively
    clustgrid <- data.frame(nk = as.integer(round(exp(seq(log(k), log(nrow(tpoints)-2),
                                                          length.out = 100)))))

    clustgrid$W <- NA
    clustgrid <- clustgrid[!duplicated(clustgrid$nk),]

    if(!is.null(prop_test)) {
      n_test <- floor(prop_test*clustgrid$nk)
      clustgrid <- clustgrid[n_test>1,]
    }

    clustgroups <- list()

    # Compute 1st PC for ordering clusters
    pcacoords <- stats::prcomp(tcoords, center = TRUE, scale. = FALSE, rank = 1)

    # We test each number of clusters
    for(nk in clustgrid$nk){

      # Create nk clusters
      if(clustering == "hierarchical"){
        clust_nk <- stats::cutree(hc, k=nk)
      }else if(clustering == "kmeans"){
        clust_nk <- stats::kmeans(tcoords, nk)$cluster
      }

      tabclust <- as.data.frame(table(clust_nk))
      tabclust$clust_k <- NA

      # compute cluster centroids and apply PC loadings to shuffle along the 1st dimension
      centr_tpoints <- sapply(tabclust$clust_nk, function(x){
        centrpca <- matrix(apply(tcoords[clust_nk %in% x, , drop=FALSE], 2, mean), nrow = 1)
        colnames(centrpca) <- colnames(tcoords)
        return(predict(pcacoords, centrpca))
      })
      tabclust$centrpca <- centr_tpoints
      tabclust <- tabclust[order(tabclust$centrpca),]

      # We don't merge big clusters
      clust_i <- 1
      for(i in 1:nrow(tabclust)){
        if(tabclust$Freq[i] >= nrow(tpoints)/k){
          tabclust$clust_k[i] <- clust_i
          clust_i <- clust_i + 1
        }
      }
      rm("clust_i")

      # And we merge the remaining into k groups
      clust_i <- setdiff(1:k, unique(tabclust$clust_k))
      if(is.null(prop_test)) {
        tabclust$clust_k[is.na(tabclust$clust_k)] <- rep(clust_i, ceiling(nk/length(clust_i)))[1:sum(is.na(tabclust$clust_k))]
      } else {
        n_train <- ceiling((1-prop_test)*nk)
        n_test <- floor(prop_test*nk)

        if(n_train > n_test) {
          v1 <- rep(1, times=n_train)
          v2 <- c(rep(2, times=n_test), rep(NA, n_train - n_test))
        } else {
          v1 <- rep(1, times=n_test)
          v2 <- c(rep(2, times=n_train), rep(NA, n_test - n_train))
        }

        v3 <- c(rbind(v1, v2))
        tabclust$clust_k[is.na(tabclust$clust_k)] <- v3[!is.na(v3)]
      }

      tabclust2 <- data.frame(ID = 1:length(clust_nk), clust_nk = clust_nk)
      tabclust2 <- merge(tabclust2, tabclust, by = "clust_nk")
      tabclust2 <- tabclust2[order(tabclust2$ID),]
      clust_k <- tabclust2$clust_k

      # Compute W statistic if not exceeding maxp
      if(!any(table(clust_k)/length(clust_k)>maxp)){

        if(isTRUE(islonglat)){
          Gjstar_i <- distclust_distmat(distmat, clust_k)
        }else{
          Gjstar_i <- distclust_euclidean(tcoords, clust_k)
        }
        clustgrid$W[clustgrid$nk==nk] <- twosamples::wass_stat(Gjstar_i, Gij)
        clustgroups[[paste0("nk", nk)]] <- clust_k
      }
    }

    # Final configuration
    k_final <- clustgrid$nk[which.min(clustgrid$W)]
    W_final <- min(clustgrid$W, na.rm=T)
    clust <- clustgroups[[paste0("nk", k_final)]]
    if(isTRUE(islonglat)){
      Gjstar <- distclust_distmat(distmat, clust)
    }else{
      Gjstar <- distclust_euclidean(tcoords, clust)
    }
  }

  # Output
  cfolds <- CAST::CreateSpacetimeFolds(data.frame(clust=clust), spacevar = "clust", k = k)
  res <- list(clusters = clust,
              indx_train = cfolds$index, indx_test = cfolds$indexOut,
              Gij = Gij, Gj = Gj, Gjstar = Gjstar,
              W = W_final, method = clustering, q = k_final, space = "geographical")
  class(res) <- c("knndm", "list")
  res
}


# kNNDM in the feature space
knndm_feature <- function(tpoints, predpoints, k, maxp, clustering, linkf, catVars, useMD, prop_test) {

  # rescale data
  if(is.null(catVars)) {

    scale_attr <- attributes(scale(tpoints))
    tpoints <- scale(tpoints) |> as.data.frame()
    predpoints <- scale(predpoints,center=scale_attr$`scaled:center`,
                        scale=scale_attr$`scaled:scale`) |>
      as.data.frame()

  } else {
    tpoints_cat <- tpoints[,catVars,drop=FALSE]
    predpoints_cat <- predpoints[,catVars,drop=FALSE]

    tpoints_num <- tpoints[,-which(names(tpoints)%in%catVars),drop=FALSE]
    predpoints_num <- predpoints[,-which(names(predpoints)%in%catVars),drop=FALSE]

    scale_attr <- attributes(scale(tpoints_num))
    tpoints <- scale(tpoints_num) |> as.data.frame()
    predpoints <- scale(predpoints_num,center=scale_attr$`scaled:center`,
                        scale=scale_attr$`scaled:scale`) |>
      as.data.frame()
    tpoints <- as.data.frame(cbind(tpoints, lapply(tpoints_cat, as.factor)))
    predpoints <- as.data.frame(cbind(predpoints, lapply(predpoints_cat, as.factor)))

  }


  # Gj and Gij calculation
  if(is.null(catVars)) {


    if(isTRUE(useMD)) {

      tpoints_mat <- as.matrix(tpoints)
      predpoints_mat <- as.matrix(predpoints)

      # use Mahalanobis distances
      if (dim(tpoints_mat)[2] == 1) {
        S <- matrix(stats::var(tpoints_mat), 1, 1)
        tpoints_mat <- as.matrix(tpoints_mat, ncol = 1)
      } else {
        S <- stats::cov(tpoints_mat)
      }
      S_inv <- MASS::ginv(S)

      # calculate distance matrix
      distmat <- matrix(nrow=nrow(tpoints), ncol=nrow(tpoints))
      distmat <- sapply(1:nrow(distmat), function(i) {
        sapply(1:nrow(distmat), function(j) {
          sqrt(t(tpoints_mat[i,] - tpoints_mat[j,]) %*% S_inv %*% (tpoints_mat[i,] - tpoints_mat[j,]))
        })
      })
      diag(distmat) <- NA

      Gj <- apply(distmat, 1, min, na.rm=TRUE)

      Gij <- sapply(1:dim(predpoints_mat)[1], function(y) {
        min(sapply(1:dim(tpoints_mat)[1], function(x) {
          sqrt(t(predpoints_mat[y,] - tpoints_mat[x,]) %*% S_inv %*% (predpoints_mat[y,] - tpoints_mat[x,]))
        }))
      })


    } else {
      # use FNN with Euclidean distances if no categorical variables are present
      Gj <- c(FNN::knn.dist(tpoints, k = 1))
      Gij <- c(FNN::knnx.dist(query = predpoints, data = tpoints, k = 1))
    }


  } else {

    # use Gower distances if categorical variables are present
    Gj <- sapply(1:nrow(tpoints), function(i) gower::gower_topn(tpoints[i,], tpoints[-i,], n=1)$distance[[1]])
    Gij <- c(gower::gower_topn(predpoints, tpoints, n = 1)$distance)

  }


  # Check if Gj > Gij (warning suppressed regarding ties)
  testks <- suppressWarnings(stats::ks.test(Gj, Gij, alternative = "great"))
  if(testks$p.value >= 0.05){

    if(!is.null(prop_test)) {
      clust <- sample(rep(1:k, ceiling(nrow(tpoints)/k)), size = nrow(tpoints), replace=T,
                      prob = rep(c(1-prop_test, prop_test), ceiling(nrow(tpoints)/k)))
    } else {
      clust <- sample(rep(1:k, ceiling(nrow(tpoints)/k)), size = nrow(tpoints), replace=F)
    }

    if(is.null(catVars)) {
      if(isTRUE(useMD)) {
        Gjstar <- distclust_MD(tpoints, clust)
      } else {
        Gjstar <- distclust_euclidean(tpoints, clust)
      }

    } else {
      Gjstar <- distclust_gower(tpoints, clust)
    }

    k_final <- "random CV"
    W_final <- twosamples::wass_stat(Gjstar, Gij)
    message("Gij <= Gj; a random CV assignment is returned")

  }else{

    if(clustering == "hierarchical"){

      # calculate distance matrix which is needed for hierarchical clustering
      if(is.null(catVars)) {

        if(isFALSE(useMD)) {
          # calculate distance matrix with Euclidean distances if no categorical variables are present
          # for MD: distance matrix was already calculated
          distmat <- stats::dist(tpoints, upper=TRUE, diag=TRUE) |> as.matrix()
          diag(distmat) <- NA
        }

      } else {

        # calculate distance matrix with Gower distances if categorical variables are present
        distmat <- matrix(nrow=nrow(tpoints), ncol=nrow(tpoints))
        for (i in 1:nrow(tpoints)){

          trainDist <-  gower::gower_dist(tpoints[i,], tpoints)

          trainDist[i] <- NA
          distmat[i,] <- trainDist
        }
      }
      hc <- stats::hclust(d = stats::as.dist(distmat), method = linkf)
    }

    # Build grid of number of clusters to try - we sample low numbers more intensively
    clustgrid <- data.frame(nk = as.integer(round(exp(seq(log(k), log(nrow(tpoints)-2),
                                                          length.out = 100)))))
    clustgrid$W <- NA
    clustgrid <- clustgrid[!duplicated(clustgrid$nk),]

    if(!is.null(prop_test)) {
      n_test <- floor(prop_test*clustgrid$nk)
      clustgrid <- clustgrid[n_test>1,]
    }

    clustgroups <- list()

    # Compute 1st PC for ordering clusters
    if(is.null(catVars)) {
      pcacoords <- stats::prcomp(tpoints, center = TRUE, scale. = FALSE, rank = 1)
    } else {
      pcacoords <- PCAmixdata::PCAmix(X.quanti = tpoints[,!(names(tpoints) %in% catVars), drop=FALSE],
                                      X.quali = tpoints[,names(tpoints) %in% catVars, drop=FALSE],
                                      graph = FALSE)
    }

    # We test each number of clusters
    for(nk in clustgrid$nk) {

      # Create nk clusters
      if(clustering == "hierarchical"){
        clust_nk <- stats::cutree(hc, k=nk)
      } else if(clustering == "kmeans"){
        if(is.null(catVars)) {
          clust_nk <- tryCatch(stats::kmeans(tpoints, nk)$cluster,
                               error=function(e) e)

        } else {
          # prototype clustering for mixed data sets
          clust_nk <- tryCatch(clustMixType::kproto(tpoints, nk,verbose=FALSE)$cluster,
                               error=function(e) e)
        }
      }

      if (!inherits(clust_nk,"error")){
        tabclust <- as.data.frame(table(clust_nk))
        tabclust$clust_k <- NA

        # compute cluster centroids and apply PC loadings to shuffle along the 1st dimension
        if(is.null(catVars)) {
          centr_tpoints <- sapply(tabclust$clust_nk, function(x){
            centrpca <- matrix(apply(tpoints[clust_nk %in% x, , drop=FALSE], 2, mean), nrow = 1)
            colnames(centrpca) <- colnames(tpoints)
            return(predict(pcacoords, centrpca))
          })
        } else {
          centr_tpoints <- sapply(tabclust$clust_nk, function(x){
            centrpca_num <- matrix(apply(tpoints[clust_nk %in% x, !(names(tpoints) %in% catVars), drop=FALSE], 2, mean), nrow = 1)
            centrpca_cat <- matrix(apply(tpoints[clust_nk %in% x, names(tpoints) %in% catVars, drop=FALSE], 2,
                                         function(y) names(which.max(table(y)))), nrow = 1)
            colnames(centrpca_num) <- colnames(tpoints[,!(names(tpoints) %in% catVars), drop=FALSE])
            colnames(centrpca_cat) <- colnames(tpoints[,names(tpoints) %in% catVars, drop=FALSE])

            return(predict(pcacoords, centrpca_num, centrpca_cat)[,1])

          })
        }

        tabclust$centrpca <- centr_tpoints
        tabclust <- tabclust[order(tabclust$centrpca),]

        # We don't merge big clusters
        clust_i <- 1
        for(i in 1:nrow(tabclust)){
          if(tabclust$Freq[i] >= nrow(tpoints)/k){
            tabclust$clust_k[i] <- clust_i
            clust_i <- clust_i + 1
          }
        }
        rm("clust_i")

        # And we merge the remaining into k groups
        clust_i <- setdiff(1:k, unique(tabclust$clust_k))

        if(is.null(prop_test)) {
          tabclust$clust_k[is.na(tabclust$clust_k)] <- rep(clust_i, ceiling(nk/length(clust_i)))[1:sum(is.na(tabclust$clust_k))]
        } else {
          n_train <- ceiling((1-prop_test)*nk)
          n_test <- floor(prop_test*nk)

          if(n_train > n_test) {
            v1 <- rep(1, times=n_train)
            v2 <- c(rep(2, times=n_test), rep(NA, n_train - n_test))
          } else {
            v1 <- rep(1, times=n_test)
            v2 <- c(rep(2, times=n_train), rep(NA, n_test - n_train))
          }

          v3 <- c(rbind(v1, v2))
          tabclust$clust_k[is.na(tabclust$clust_k)] <- v3[!is.na(v3)]
        }

        tabclust2 <- data.frame(ID = 1:length(clust_nk), clust_nk = clust_nk)
        tabclust2 <- merge(tabclust2, tabclust, by = "clust_nk")
        tabclust2 <- tabclust2[order(tabclust2$ID),]
        clust_k <- tabclust2$clust_k



        # Compute W statistic if not exceeding maxp
        if(!(any(table(clust_k)/length(clust_k)>maxp))){

          if(clustering == "kmeans") {
            if(is.null(catVars)) {
              if(isTRUE(useMD)){
                Gjstar_i <- distclust_MD(tpoints, clust_k)
              } else {
                Gjstar_i <- distclust_euclidean(tpoints, clust_k)
              }
            } else {
              Gjstar_i <- distclust_gower(tpoints, clust_k)
            }

          } else {
            Gjstar_i <- distclust_distmat(distmat, clust_k)
          }

          clustgrid$W[clustgrid$nk==nk] <- twosamples::wass_stat(Gjstar_i, Gij)
          clustgroups[[paste0("nk", nk)]] <- clust_k
        }
      } else {
        message(paste("skipped nk", nk))
      }
    }

    # Final configuration
    k_final <- clustgrid$nk[which.min(clustgrid$W)]
    W_final <- min(clustgrid$W, na.rm=T)
    clust <- clustgroups[[paste0("nk", k_final)]]

    if(clustering == "kmeans") {
      if(is.null(catVars)) {
        if(isTRUE(useMD)) {
          Gjstar <- distclust_MD(tpoints, clust)
        } else {
          Gjstar <- distclust_euclidean(tpoints, clust)
        }

      } else {
        Gjstar <- distclust_gower(tpoints, clust)
      }
    } else {
      Gjstar <- distclust_distmat(distmat, clust)
    }

  }


  # Output
  cfolds <- CAST::CreateSpacetimeFolds(data.frame(clust=clust), spacevar = "clust", k = k)
  res <- list(clusters = clust,
              indx_train = cfolds$index, indx_test = cfolds$indexOut,
              Gij = Gij, Gj = Gj, Gjstar = Gjstar,
              W = W_final, method = clustering, q = k_final, space = "feature")
  class(res) <- c("knndm", "list")
  res
}

# kNNDM in both, feature and geographical space
knndm_feature_geo <- function(tpoints, predpoints, k, maxp, clustering, linkf,
                              islonglat, catVars, useMD, prop_test, lambda) {



  tpoints_df <- sf::st_drop_geometry(tpoints)
  predpoints_df <- sf::st_drop_geometry(predpoints)

  # rescale data
  if(is.null(catVars)) {

    scale_attr <- attributes(scale(tpoints_df))
    tpoints_df <- scale(tpoints_df) |> as.data.frame()
    predpoints_df <- scale(predpoints_df,center=scale_attr$`scaled:center`,
                        scale=scale_attr$`scaled:scale`) |>
      as.data.frame()

  } else {
    tpoints_df_cat <- tpoints_df[,catVars,drop=FALSE]
    predpoints_df_cat <- predpoints_df[,catVars,drop=FALSE]

    tpoints_df_num <- tpoints_df[,-which(names(tpoints_df)%in%catVars),drop=FALSE]
    predpoints_df_num <- predpoints_df[,-which(names(predpoints_df)%in%catVars),drop=FALSE]

    scale_attr <- attributes(scale(tpoints_df_num))
    tpoints_df <- scale(tpoints_df_num) |> as.data.frame()
    predpoints_df <- scale(predpoints_df_num,center=scale_attr$`scaled:center`,
                        scale=scale_attr$`scaled:scale`) |>
      as.data.frame()
    tpoints_df <- as.data.frame(cbind(tpoints_df, lapply(tpoints_df_cat, as.factor)))
    predpoints_df <- as.data.frame(cbind(predpoints_df, lapply(predpoints_df_cat, as.factor)))

  }


  # Gj and Gij calculation in geographical space
  tcoords <- sf::st_coordinates(tpoints)[,1:2]
  if(isTRUE(islonglat)){
    distmat <- sf::st_distance(tpoints)
    units(distmat) <- NULL
    diag(distmat) <- NA
    Gj_geo <- apply(distmat, 1, function(x) min(x, na.rm=TRUE))
    Gij_geo <- sf::st_distance(predpoints, tpoints)
    units(Gij_geo) <- NULL
    Gij_geo <- apply(Gij_geo, 1, min)
  }else{
    Gj_geo <- c(FNN::knn.dist(tcoords, k = 1))
    Gij_geo <- c(FNN::knnx.dist(query = sf::st_coordinates(predpoints)[,1:2],
                            data = tcoords, k = 1))
  }

  # Gj_feature and Gij_feature calculation in feature space
  if(is.null(catVars)) {

    if(isTRUE(useMD)) {

      tpoints_df_mat <- as.matrix(tpoints_df)
      predpoints_df_mat <- as.matrix(predpoints_df)

      # use Mahalanobis distances
      if (dim(tpoints_df_mat)[2] == 1) {
        S <- matrix(stats::var(tpoints_df_mat), 1, 1)
        tpoints_df_mat <- as.matrix(tpoints_df_mat, ncol = 1)
      } else {
        S <- stats::cov(tpoints_df_mat)
      }
      S_inv <- MASS::ginv(S)

      # calculate distance matrix
      distmat <- matrix(nrow=nrow(tpoints_df), ncol=nrow(tpoints_df))
      distmat <- sapply(1:nrow(distmat), function(i) {
        sapply(1:nrow(distmat), function(j) {
          sqrt(t(tpoints_df_mat[i,] - tpoints_df_mat[j,]) %*% S_inv %*% (tpoints_df_mat[i,] - tpoints_df_mat[j,]))
        })
      })
      diag(distmat) <- NA

      Gj_feature <- apply(distmat, 1, min, na.rm=TRUE)

      Gij_feature <- sapply(1:dim(predpoints_df_mat)[1], function(y) {
        min(sapply(1:dim(tpoints_df_mat)[1], function(x) {
          sqrt(t(predpoints_df_mat[y,] - tpoints_df_mat[x,]) %*% S_inv %*% (predpoints_df_mat[y,] - tpoints_df_mat[x,]))
        }))
      })


    } else {
      # use FNN with Euclidean distances if no categorical variables are present
      Gj_feature <- c(FNN::knn.dist(tpoints_df, k = 1))
      Gij_feature <- c(FNN::knnx.dist(query = predpoints_df, data = tpoints_df, k = 1))
    }


  } else {

    # use Gower distances if categorical variables are present
    Gj_feature <- sapply(1:nrow(tpoints_df), function(i) gower::gower_topn(tpoints_df[i,], tpoints_df[-i,], n=1)$distance[[1]])
    Gij_feature <- c(gower::gower_topn(predpoints_df, tpoints_df, n = 1)$distance)

  }


  # scale geographical and feature-space NNDs
  range01 <- function(x){(x-min(x))/(max(x)-min(x))}
  NNDs_stand <- lapply(list("Gj_feature"=Gj_feature, "Gj_geo"=Gj_geo, "Gij_feature"=Gij_feature,
                            "Gij_geo"=Gij_geo), range01)

  Gj <- NNDs_stand$Gj_geo + lambda*NNDs_stand$Gj_feature
  Gij <- NNDs_stand$Gij_geo + lambda*NNDs_stand$Gij_feature


  # Check if Gj_feature > Gij_feature | Gj_geo > Gij_geo (warning suppressed regarding ties)
  testks_geo <- suppressWarnings(stats::ks.test(Gj_feature, Gij_feature, alternative = "great"))
  testks_feature <- suppressWarnings(stats::ks.test(Gj_geo, Gij_geo, alternative = "great"))
  if(testks_geo$p.value >= 0.05 | testks_feature$p.value >= 0.05){

    if(!is.null(prop_test)) {
      clust <- sample(rep(1:k, ceiling(nrow(tpoints_df)/k)), size = nrow(tpoints_df), replace=T,
                      prob = rep(c(1-prop_test, prop_test), ceiling(nrow(tpoints_df)/k)))
    } else {
      clust <- sample(rep(1:k, ceiling(nrow(tpoints_df)/k)), size = nrow(tpoints_df), replace=F)
    }

    if(is.null(catVars)) {
      if(isTRUE(useMD)) {
        Gj_featurestar <- distclust_MD(tpoints_df, clust)
      } else {
        Gj_featurestar <- distclust_euclidean(tpoints_df, clust)
      }

    } else {
      Gj_featurestar <- distclust_gower(tpoints_df, clust)
    }

    k_final <- "random CV"
    W_final <- twosamples::wass_stat(Gj_featurestar, Gij_feature)
    message("Gij_feature <= Gj_feature; a random CV assignment is returned")

  }else{


    # geographical similarities
    distmat_geo <- apcluster::negDistMat(tcoords, r = 1) #2?

    # feature similarities
    distmat_feature <- apcluster::negDistMat(tpoints_df, r = 1)

    # create assymetric distance matrix with feature space distance in upper triangle
    distmat <- distmat_feature

    # paste geographical similarities in lower triangle
    distmat[lower.tri(distmat)] <- distmat_geo[lower.tri(distmat_geo)]

    # Build grid of number of clusters to try - we sample low numbers more intensively
    clustgrid <- data.frame(nk = as.integer(round(exp(seq(log(k), log(nrow(tpoints)-2),
                                                          length.out = 100)))))
    clustgrid$W <- NA
    clustgrid <- clustgrid[!duplicated(clustgrid$nk),]

    clustgroups <- list()

    # Compute 1st PC for ordering clusters
    if(is.null(catVars)) {
      pcacoords <- stats::prcomp(tpoints_df, center = TRUE, scale. = FALSE, rank = 1)
    } else {
      pcacoords <- PCAmixdata::PCAmix(X.quanti = tpoints_df[,!(names(tpoints_df) %in% catVars), drop=FALSE],
                                      X.quali = tpoints_df[,names(tpoints_df) %in% catVars, drop=FALSE],
                                      graph = FALSE)
    }

    # We test each number of clusters
    for(nk in clustgrid$nk) {

      print(nk)

      # Create nk clusters by affinity propagation
      clust_nk <- apcluster::aggExCluster(distmat)
      clust_nk <- apcluster::cutree(clust_nk, k=nk)@idx |> as.vector()

      if (!inherits(clust_nk,"error")){
        tabclust <- as.data.frame(table(clust_nk))
        tabclust$clust_k <- NA

        # compute cluster centroids and apply PC loadings to shuffle along the 1st dimension
        if(is.null(catVars)) {
          centr_tpoints_df <- sapply(tabclust$clust_nk, function(x){
            centrpca <- matrix(apply(tpoints_df[clust_nk %in% x, , drop=FALSE], 2, mean), nrow = 1)
            colnames(centrpca) <- colnames(tpoints_df)
            return(predict(pcacoords, centrpca))
          })
        } else {
          centr_tpoints_df <- sapply(tabclust$clust_nk, function(x){
            centrpca_num <- matrix(apply(tpoints_df[clust_nk %in% x, !(names(tpoints_df) %in% catVars), drop=FALSE], 2, mean), nrow = 1)
            centrpca_cat <- matrix(apply(tpoints_df[clust_nk %in% x, names(tpoints_df) %in% catVars, drop=FALSE], 2,
                                         function(y) names(which.max(table(y)))), nrow = 1)
            colnames(centrpca_num) <- colnames(tpoints_df[,!(names(tpoints_df) %in% catVars), drop=FALSE])
            colnames(centrpca_cat) <- colnames(tpoints_df[,names(tpoints_df) %in% catVars, drop=FALSE])

            return(predict(pcacoords, centrpca_num, centrpca_cat)[,1])

          })
        }

        tabclust$centrpca <- centr_tpoints_df
        tabclust <- tabclust[order(tabclust$centrpca),]

        # We don't merge big clusters
        clust_i <- 1
        for(i in 1:nrow(tabclust)){
          if(tabclust$Freq[i] >= nrow(tpoints_df)/k){
            tabclust$clust_k[i] <- clust_i
            clust_i <- clust_i + 1
          }
        }
        rm("clust_i")

        # And we merge the remaining into k groups
        clust_i <- setdiff(1:k, unique(tabclust$clust_k))

        tabclust$clust_k[is.na(tabclust$clust_k)] <- rep(clust_i, ceiling(nk/length(clust_i)))[1:sum(is.na(tabclust$clust_k))]

        tabclust2 <- data.frame(ID = 1:length(clust_nk), clust_nk = clust_nk)
        tabclust2 <- merge(tabclust2, tabclust, by = "clust_nk")
        tabclust2 <- tabclust2[order(tabclust2$ID),]
        clust_k <- tabclust2$clust_k

        # Compute W statistic if not exceeding maxp
        if(!(any(table(clust_k)/length(clust_k)>maxp))){

          diag(distmat) <- NA
          Gjstar_i <- distclust_distmat(distmat*(-1), clust_k)

          clustgrid$W[clustgrid$nk==nk] <- twosamples::wass_stat(Gjstar_i, Gij)
          clustgroups[[paste0("nk", nk)]] <- clust_k
        }
      } else {
        message(paste("skipped nk", nk))
      }
    }

    # Final configuration
    k_final <- clustgrid$nk[which.min(clustgrid$W)]
    W_final <- min(clustgrid$W, na.rm=T)
    clust <- clustgroups[[paste0("nk", k_final)]]

    if(clustering == "kmeans") {
      if(is.null(catVars)) {
        if(isTRUE(useMD)) {
          Gj_featurestar <- distclust_MD(tpoints, clust)
        } else {
          Gj_featurestar <- distclust_euclidean(tpoints, clust)
        }

      } else {
        Gj_featurestar <- distclust_gower(tpoints, clust)
      }
    } else {
      Gj_featurestar <- distclust_distmat(distmat*(-1), clust)
    }

  }


  # Output
  cfolds <- CAST::CreateSpacetimeFolds(data.frame(clust=clust), spacevar = "clust", k = k)
  res <- list(clusters = clust,
              indx_train = cfolds$index, indx_test = cfolds$indexOut,
              Gij = Gij_feature, Gj = Gj_feature, Gjstar = Gj_featurestar,
              W = W_final, method = clustering, q = k_final, space = "feature")
  class(res) <- c("knndm", "list")
  res
}


# Helper function: Compute out-of-fold NN distance (geographical coordinates / numerical variables)
distclust_distmat <- function(distm, folds){
  alldist <- rep(NA, length(folds))
  for(f in unique(folds)){
    alldist[f == folds] <- apply(distm[f == folds, f != folds, drop=FALSE], 1, min)
  }
  alldist
}

# Helper function: Compute out-of-fold NN distance (projected coordinates / numerical variables)
distclust_euclidean <- function(tr_coords, folds){
  alldist <- rep(NA, length(folds))
  for(f in unique(folds)){
    alldist[f == folds] <- c(FNN::knnx.dist(query = tr_coords[f == folds,,drop=FALSE],
                                            data = tr_coords[f != folds,,drop=FALSE], k = 1))
  }
  alldist
}

# Helper function: Compute out-of-fold NN distance (categorical variables)
distclust_gower <- function(tr_coords, folds){

  alldist <- rep(NA, length(folds))
  for(f in unique(folds)){
    alldist[f == folds] <- c(gower::gower_topn(tr_coords[f == folds,,drop=FALSE],
                                               tr_coords[f != folds,,drop=FALSE], n=1))$distance[[1]]
  }
  unlist(alldist)
}

# Helper function: Compute out-of-fold NN distance (Mahalanobian distance)
distclust_MD <- function(tr_coords, folds){

  tr_mat <- as.matrix(tr_coords)

  S <- stats::cov(tr_mat)
  S_inv <- MASS::ginv(S)

  alldist <- rep(NA, length(folds))
  for(f in unique(folds)) {

    alldist[f == folds] <- apply(tr_mat[f==folds,,drop=FALSE], 1, function(y) {
      min(apply(tr_mat[f!=folds,,drop=FALSE], 1, function(x) {
        sqrt(t(y - x) %*% S_inv %*% (y - x))
      }))
    })
  }
  unlist(alldist)
}
