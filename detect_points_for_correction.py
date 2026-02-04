def detect_points_for_correction(folder, wavelength_list):
    
    import os
    import numpy as np
    from detect_n_objects import detect_n_objects

    if os.path.exists(folder + r'\image' + str(wavelength_list[0]) + '.jpg'):
        ext = '.jpg'
    else:
        ext = '.png'

    points = []
    n = len(wavelength_list)
    i = 0

    for wavelength in wavelength_list:
        detections, image = detect_n_objects(folder + r'\image' + str(wavelength) + ext, 100)
        points.append([wavelength, detections])
        print(f"\rDetecting grains on images: {i+1}/{n}", end="", flush=True)
        i += 1


    joint_array = np.vstack([np.column_stack((np.full(arr.shape[0], i), arr)) for i, arr in points])
    result = joint_array[(joint_array[:, -1] == 0) | (joint_array[:, -1] == 2)]
    print("\nFolder done: " + folder)

    return(result)


if __name__ == '__main__':

    main_folder = r'grain_11.12.2025_B39-1-1_ostatok7.68'
    wavelength_list = list(range(400,1001,10))

    points_for_correcrion = detect_points_for_correction(folder=main_folder + r'\Spectral_Cube', wavelength_list=wavelength_list)
    print(points_for_correcrion)
