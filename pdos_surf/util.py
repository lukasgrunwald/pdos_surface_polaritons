import sys
import os

CM = 1/2.54

def use_style(name):
    """ Applies a custom Matplotlib style globally in the main script.

    This function constructs a path to a `.mplstyle` file stored in the user's
    Matplotlib configuration directory ~/.matplotlib and applies it as the current
    plotting style. The style is applied in the global scope of the main script using `exec`.

    Parameters:
    -----------
    name : str
        The name of the `.mplstyle` file (without extension) located in
        '/Users/grunwal/.matplotlib/'.

    Notes:
    ------
    - This function assumes the `.mplstyle` file exists in the specified directory.
    - The `exec` function is used to apply the style within the global scope of
      the main script, which may introduce security risks if used improperly.

    Example:
    --------
    ```python
    use_style("dark_theme")  # Applies '/Users/grunwal/.matplotlib/dark_theme.mplstyle'
    ```
    """
    path = f"'/Users/grunwal/.matplotlib/{name}.mplstyle'"
    code = f"plt.style.use({path})"

    main_globals = sys.modules['__main__'].__dict__  # Get globals of the main script
    exec(code, main_globals)  # Execute in main script's scope


def versionized_path(path, overwrite = False):
    """
    Checks if a FILE exists and appends '_vN' until a unique name is found.
    """
    if not os.path.exists(path) or overwrite:
        return path

    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    base, ext = os.path.splitext(filename)

    version = 2
    for i in range(200): # Maximal file Number is 200
        versioned_filename = f"{base}_v{version}{ext}"
        new_full_path = os.path.join(directory, versioned_filename)

        if not os.path.exists(new_full_path):
            return new_full_path

        version += 1

    raise FileExistsError("Couldn't fine a unique file name for {path}")
