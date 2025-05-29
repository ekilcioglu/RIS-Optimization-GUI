Finding RIS Target Points via K-means Clustering
################################################

This tutorial explains how to find a specific number of RIS target points by applying K-means clustering algorithm to the low-power cells in the transmitter-only coverage map.

.. note::

   Before executing this step, make sure to first compute and visualize the transmitter-only coverage map. This is necessary to extract the coordinates of low-power cells.  
   Please follow the `Computing Transmitter-Only Coverage Map` tutorial before proceeding.

**Steps to Perform K-means Clustering:**

1. **Set Number of Target Points**

   In the GUI, go to the labelframe **"Manual trials"** on the left side. Enter the desired number of RIS target points in the field labeled **"Number of target points"**.

2. **Run Clustering Algorithm**

   Click the button **"Find the target point(s) via clustering algorithm"**. This will apply K-means clustering on the low-power cells and determine the optimal RIS target points.

3. **View the Results**

   After execution, a **binary poor coverage map** is displayed. The selected RIS target points will be marked with **green 'X' symbols** on the map.  
   An example output is shown below. Also, the coordinates of the selected target points are demonstrated under the labelframe **"Messages"**.

.. figure:: finding_target_points_via_clustering_Fig1.png
   :align: center
   :figwidth: 80%
   :name: finding_target_points_via_clustering_Fig1

   **Fig. 1**: Binary poor coverage map with RIS target points marked as green 'X'