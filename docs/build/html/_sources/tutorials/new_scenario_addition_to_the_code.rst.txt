New Scenario Addition to the Source Code
########################################

This tutorial explains how to add a newly created scenario to the source code of our RIS optimization tool.

.. note::

  Before executing this step, you must first create a new scenario in Blender and export it as a `.xml` file.  
  Please follow the `New Scenario Creation in Blender` tutorial beforehand.

1. **Update the Scenario List**

   - Search for **`self.scenario_options`** in the source code.
   - Add a new entry to the list with the name of your new scenario.

2. **Modify the Scenario Variable (`self.scenario_var`)**

   Search for **`self.scenario_var`** and update the following methods:

   - **In `load_scenario()`**:  
     Add an additional `elif` statement for your new scenario. Follow the structure used for previous scenarios. You can use the template below:

     .. code-block:: python

         elif self.scenario_var.get() == "New_Scenario_Name":
             self.scene = load_scene("Scenarios/New_Scenario_Folder_Name/New_Scenario_Name.xml")
             # Indicating the wall indices in the coverage map
             self.zero_indices = [ ]  # Fill this list with indices where walls are located!
             # Define walls for RIS placement
             self.RIS_search_positions = [
                 {"fixed_coord": "x", "fixed_value": 0.1, "variable_coord": "y", "min": 0.1, "max": 15.9},
                 {"fixed_coord": "y", "fixed_value": 0.1, "variable_coord": "x", "min": 0.1, "max": 3.9},
             ]  # Adjust this dictionary to define valid RIS search positions.

   - **In `set_preset_values()`** *(Optional)*:  
     Add another `elif` statement to preset the transmitter coordinates for the new scenario:

     .. code-block:: python

         elif scenario == "New_Scenario_Name":
             set_tx_coordinates(3.6, 15.9, 2)  # Set the TX coordinates for the new scenario!

   - **In `preview_scenario()`** *(Optional)*:  
     Add another `elif` statement to set up **camera positions** for previewing the new scenario in the GUI.  
     This helps visually verify the placements of the transmitter and RIS.

**Important Reminders:**

- **zero_indices** must accurately reflect where the walls are in your new scenario.
- **RIS_search_positions** should correctly define the candidate walls where the RIS can be placed.
- Adjust the camera positions carefully in `preview_scenario()` for a better visual verification.
- If there are some gaps in the coverage maps for the newly created scenario, you can increase the number of samples of the coverage map plots by following the procedure below:

  - Search for **`self.scene.coverage_map(`** in the source code, where you will find 4 matches.
  - You can increase **`num_samples`** value of each match to obtain better coverage maps with the cost of longer simulation time.

