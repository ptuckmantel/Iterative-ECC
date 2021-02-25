# Iterative-ECC
Iterative ECC image shift correction for AFM images with smooth topography with extended features

Context: 
Atomic force microscopy imaging suffers from distortions and shifts due to instrument artefacts. These artefacts need to be corrected when sequential images of the same area are acquired and parameters are to be tracked across the time series. Correction of these distortions can be performed through keypoint detection, by finding corresponding keypoints in the topography images of a reference scan (typically the first in the series) and target scans. The relative shifts in positions of these keypoints allows the distortion to be quantified and the target image can be corrected. However keypoint detection and matching works less well when the topography contrasts are smooth and the topographical features are extended across the image, rather than localised blobs.

In these cases, ECC correction works better but only for small shifts. ECC correction maximises a correlation function between reference and target images while varying the parameters of the warp matrix describing the transformation between reference and target image. As it is an optimisation problem, it requires an initial warp matrix. Typically an identity matrix is used, yielding good results for small distortions. However when image series display a wide range of distortions, this approach does not work well.

Iterative ECC offers a simple solution by trying multiple starting input matrices describing shifts on a grid centered on the reference image and choosing the resulting corrected image offering the best correction. This can be done iteratively to gradually find the best correction. This reduces the number of attempts that need to be performed.
