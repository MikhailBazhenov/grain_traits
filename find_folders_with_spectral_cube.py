import os

def find_folders_with_spectral_cube(main_folder: str) -> list[str]:
    """
    Search all immediate subfolders of `main_folder` and return those
    that contain a 'Spectral_Cube' directory.
    """
    valid_folders = []

    # Iterate through items in the main folder
    for item in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, item)

        # Check if the item is a directory
        if os.path.isdir(subfolder_path):
            spectral_cube_path = os.path.join(subfolder_path, "Spectral_Cube")

            # Check for presence of Spectral_Cube folder
            if os.path.isdir(spectral_cube_path):
                valid_folders.append(item)

    return valid_folders
