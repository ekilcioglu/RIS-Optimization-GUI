Running RIS Joint Optimization Algorithm and Drawing Performance Evaluation Plots
#################################################################################

This tutorial explains how to run the RIS joint optimization algorithm and how to draw performance evaluation plots after obtaining all performance metric results of all possible configurations.

.. note::

   Before executing this step, you must first compute and visualize the transmitter-only coverage map.  
   Please follow the `Computing Transmitter-Only Coverage Map` tutorial beforehand.

1. **Choose Phase Profile Approach**

   - Select the desired phase profile approach from the dropdown next to the textlabel **"Choose phase profile approach"**.
   - If **"Manual entry"** is selected:

      - A new menu  appears at the bottom of the GUI under the labelframe **"Select manual phase profile file (.json)"**.
      - Click the **"Browse"** button to select the phase profile `.json` file.

2. **Enter the Parameter Boundaries**

   - Enter the lowest and highest number of target points, along with the step size for the search, under the labelframe **"Number of target point interval (N_lower, N_upper, N_step)"**, which is located under **"Optimization algorithm"**.
   - Enter the RIS height under **"RIS height (m)"**.
   - Enter the lowest and highest RIS widths, along with the step size for the search, under **"RIS width interval (m) (W_RIS_lower, W_RIS_upper, W_RIS_step)"**.
   - Choose a metric computation technique under the labelframe **"Metric computation technique"**:

      - If **"Using coverage map"** is chosen, the algorithm computes the full coverage map for each RIS configuration (more accurate but slower).
      - If **"Using individual path computation"** is chosen, the algorithm only computes the power levels at previously identified low-power cells (faster).

3. **Compute performance metrics**

   Click the button **"Compute performance metrics for all N, W_RIS, r_RIS"** to compute all performance metrics across all RIS configurations, including variations in the number of target points, RIS widths, and feasible RIS positions.

   .. important::

      This operation can take a considerable amount of time.  
      Please do not close the program until the operation is completed or you see an error in the Python interpreter or under the labelframe **"Messages"**!

4. **Draw Performance Evaluation Plots**

   At the end of Step 3, all performance metric results along with the corresponding RIS configurations will be automatically exported as a `.json` file in the root directory of the RIS optimization framework's source code.

   To visualize the best configuration considering a performance improvement threshold:

   - Select the `.json` data file under the labelframe **"Select data file (.json)"** by clicking the **"Browse"** button.
   - Enter a performance improvement threshold (dB) value under the labelframe **"Performance improvement threshold (dB)"**. This threshold defines the minimum performance gain required to justify increasing the RIS width.
   - Click the button **"Show sub-optimal optimization parameters (N^opt, W_RIS^opt, r^opt)"** to visualize the results.

An example of the resulting figure is shown below:

.. figure:: running_opt_algorithm_Fig1.png
   :align: center
   :figwidth: 100%
   :name: running_opt_algorithm_Fig1

   **Fig. 1**: Performance evaluation plot

