# Writeup: Track 3D-Objects Over Time

Please use this starter template to answer the following questions:

### 1. Write a short recap of the four tracking steps and what you implemented there (filter, track management, association, camera fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?

All my results are stored under the directory *./my_results*.

I implemented the following steps which were required:

* in step 1 *filter.py* (Kalman filter):
  * System matrix F and process noise covariance Q functions,
  * The track predict and update functions.

* in step 2 *trackmanagement.py* (Track management):
  * in track object the initialisation step is done based on an unassigned lidar measurement object,
  * in track management the assignment of unassigned measuremnts and
  * in track management the maintenance of track score and state was implemented plus the deletion of tracks

* in step 3 *association.py* (Association):
  * the association matrix was implemented with the help of Mahalanobis distances and gating functions
  * and get_closest_track_and_meas() functions for maintaining the assocoiation matrix and the unassigned track and mesurement lists

  Smalest RMSE mean: 0.12, hihghest RMSE mean: 0.19

* in step 4 *measuremnt.py* (sensor fusion):
  * the *in field of view* (in_fov())function was implemented,
  * the get_hx() function for handling non linear camera measurements,
  * in the Measuremnt class the __z__ vector and __R__ matrix is initialized for camera measurements,

  Smalest RMSE mean: 0.1, hihghest RMSE mean: 0.17

### 2. Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)?

In theory sensor fusion have more benefits:

* add redundancy which results more reliable tracks,
* the accuracy of the tracks can be increased.

The results shows between step 3 and step 4 that the mean RMSE is increasing.

### 3. Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?

More sensors results more computations so at the end the hardwere will be more expensive or the process will be slower on the same hardware.
In step 4 I realized in the project that my configuration did not worked with GPU and so the processing took more time.

### 4. Can you think of ways to improve your tracking results in the future?

One way is to use better sensors (lidar, camera). Of course better models for both lidar and camera can improve accuracy.
