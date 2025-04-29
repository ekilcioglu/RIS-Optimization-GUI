New Scenario Creation in Blender
################################

This tutorial explains how to create a new 3D scenario in Blender and make it compatible with our RIS optimization tool and Sionna ray-tracing tool.

.. note::

   Please use **Blender 3.6** or newer versions for scenario creation.


1. **Design a 3D Scenario in Blender**

   - Open Blender and design your indoor 3D environment (e.g., walls, floors, ceilings).
   - To assist with creating indoor structures like walls, you can use the **Archimesh add-on**.

   .. note::

      For details about the Archimesh add-on, visit:  
      https://extensions.blender.org/add-ons/archimesh/


2. **Define Compatible Material Names**

   For the scenario to be compatible with Sionna ray-tracing tool, after creating walls, ceilings, and floors, assign radio material names that are allowed by Sionna.

   To assign material names:

   - Select each object (e.g., a wall) in Blender.
   - Go to the **Material** tab.
   - Change the material name to a valid radio material name.

   .. note::

      Refer to the list of Sionna-compatible radio material names here:  
      https://nvlabs.github.io/sionna/rt/api/radio_materials.html

   .. important::

      If you are using **Sionna version 0.19.2** (which our RIS optimization tool uses),  
      you must add the prefix **"itu_"** to the material name.

      For example:
      
      - Wall material: **itu_plasterboard**
      - Floor material: **itu_chipboard**

3. **Export the Scenario to Mitsuba XML Format**

   After assigning correct material names:

   - Click *File → Export → Mitsuba (.xml)*.
   - In the export settings on the right side:

      - Set **Forward** to **Y Forward**.
      - Set **Up** to **Z Up**.
      - Check the box **"Export IDs"**.

   - Enter the desired filename for the `.xml` file.
   - Click **"Mitsuba Export"** to save the file.

   .. note::

      If all materials are correctly named and the export settings are properly configured,  
      the generated `.xml` file will be compatible with both our RIS optimization tool and Sionna ray-tracing tool.
