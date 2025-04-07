Computing combined coverage map with custom RIS parameter specifications
########################################################################

This section explains how to compute a combined coverage map, considering the contributions of both the transmitter and the placed RIS.

.. note::

   Before executing this step, you must first compute and visualize the **transmitter-only coverage map**.  
   Please follow the `Computing Transmitter-Only Coverage Map` tutorial beforehand.

Step 1: **Define RIS Target Points**

There are two ways to define the RIS target points:

1. **Using the Target Points from Clustering**

.. note::

   To use this option, you must first run the clustering algorithm to compute target points.  
   Refer to the `Finding RIS Target Points via K-means Clustering` tutorial before proceeding.

In the GUI, select the radio button **"Use the target point(s) found via clustering algorithm"**.

2. **Manually Entering Target Point Coordinates**

- Go to the labelframe **"Manual trials"** on the left side of the GUI.
- Enter the number of RIS target points in the field **"Number of target points"**
- Select the checkbox **"Enter the target point(s) manually"**.
- A new input area will appear at the bottom of the same labelframe.
- Enter the **x, y, z coordinates** for each target point manually.

Step 2: **Enter RIS Parameters**

- Set the RIS center position under the labelframe **"Enter RIS center position (m) (x,y,z)"**.
- Set the RIS height and width under **"RIS height (m)"** and **"RIS width (m)"**, respectively.

.. note::

   To determine feasible RIS positions in the scene, refer to the `Computing Feasible RIS Positions` tutorial.

Step 3: **Choose Phase Profile Approach**

- Select the desired phase profile approach from the dropdown next to the textlabel **"Choose phase profile approach"**.
- If **"Manual entry"** is selected:
  - A new menu  appears at the bottom of the GUI under the labelframe **"Select manual phase profile file (.json)"**.
  - Click the **"Browse"** button to select the phase profile `.json` file.

Step 4: **Computing combined coverage map**
