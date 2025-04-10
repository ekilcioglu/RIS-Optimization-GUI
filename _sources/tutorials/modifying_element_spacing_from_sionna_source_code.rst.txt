Modifying RIS Element Spacing (In Proportion to the Wavelength λ) from Sionna Source Code
###########################################################################################

This section explains how to modify the **RIS element spacing** from the default value of λ/2 to a custom value proportional to the wavelength λ by directly editing the Sionna source code.

.. note::

   The Sionna ray-tracing tool does not provide a direct way to modify the RIS element spacing via its standard interface.  
   However, it can be changed by manually editing the source code.

.. warning::

   Before making any changes, it's important to create a backup of the original `ris.py` file.  
   This allows you to easily restore it if needed.

1. **Locate the `ris.py` File**

- Navigate to the following path where the Sionna ray-tracing source files are located:

  ``C:\Users\your_user_name\AppData\Local\Programs\Python\Python311\Lib\site-packages\sionna\rt``

- Open the file **`ris.py`** for editing.

2. **Modify the RIS Element Spacing**

Apply the following changes inside the `ris.py` file:

- **Search for**:

   ``wavelength/tf.cast(2,``

   (You should find three matches.)

   **Replace all occurrences with**:

   ``wavelength/tf.cast(x,``

   where **x** is the new denominator so that the element spacing becomes **λ/x** instead of **λ/2**.

- **Search for**:

   ``tf.cast(0.5, self._rdtype)``

   (You should find three matches.)

   **Replace all occurrences with**:

   ``tf.cast(1/x, self._rdtype)``

.. note::

   Set **x** to your desired value.  
   For example, to set the element spacing to λ/3, use **x = 3**.

3. **Save and Restart**

- Save the modified `ris.py` file.
- Restart your Python environment to ensure that the changes take effect.