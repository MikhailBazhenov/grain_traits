

def sd_to_cv(df, wavelength_list):
    
    columns_to_drop = []

    for wave in wavelength_list:
        df['CV_' + str(wave)] = df['sd_' + str(wave)] / df['mean_' + str(wave)]
        columns_to_drop.append('sd_' + str(wave))

    df = df.drop(columns=columns_to_drop)
    
    return df


def get_data_from_plate(
        folder = r'grain_11.12.2025_empty_plate',
        wavelength_list = list(range(400,1001,10)),
        empty=True
        ):
    
    # Импортируем все нужные функции из файлов 

    import cv2
    import pandas as pd
    from matplotlib import pyplot as plt

    from build_df_spectra_from_masks import build_df_spectra_from_masks
    from build_wave_metrics_numpy import build_wave_metrics_numpy
    from compute_mask_metrics import compute_mask_metrics
    from detect_n_objects import detect_n_objects
    from detect_points_for_correction import detect_points_for_correction
    from make_comparison_video import make_comparison_video
    from rectify_folder_no_crop_resize_to_original import rectify_folder_no_crop_resize_to_original
    from run_batch_alignment import run_batch_alignment
    from segment_with_sam_bbox import segment_with_sam_bbox
    from sort_detection_boxes import sort_detection_boxes
    from filter_cells_by_global_mean_percentiles import filter_cells_by_global_mean_percentiles
    from scale_means_per_cell import scale_means_per_cell


    # Детектируем точки для коррекции на снимках пустого планшета

    
    
    points_for_correcrion_empty = detect_points_for_correction(folder=folder + r'\Spectral_Cube', wavelength_list=wavelength_list)

    df_points_empty = pd.DataFrame(points_for_correcrion_empty)
    df_points_empty.columns = ['wave', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'cls']
    df_points_empty['x'] = (df_points_empty['x_min'] + df_points_empty['x_max']) / 2
    df_points_empty['y'] = (df_points_empty['y_min'] + df_points_empty['y_max']) / 2
    df_points_empty_new = df_points_empty[['wave', 'x', 'y']]

    # Собираем информацию для коррекции по точкам

    df_metrics_empty = build_wave_metrics_numpy(df_points_empty_new, wavelength_list, base_wave=1000, chunk_size=5000)

    # Проводим выравнивание снимков пустого планшета
    print('Making alignment of images...')

    out_dir_empty = run_batch_alignment(
            df=df_metrics_empty,
            main_folder=folder,
            subfolder="Spectral_Cube",
            out_subfolder="Spectral_Cube_Processed",
            ext_priority=(".png", ".jpg", ".jpeg"),
            keep_alpha_if_png=True,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=(0, 0, 0),  # fill color for newly exposed areas
        )

    print(f"Done. Aligned images are in: {out_dir_empty}")

    # Устраняем перспективу
    print('Making rectification of images...')
    df_angle_correction_empty = df_points_empty_new[df_points_empty_new['wave'] == 1000].drop('wave', axis=1).reset_index(drop=True)

    out_dir, rect_w_px, rect_h_px = rectify_folder_no_crop_resize_to_original(
            main_folder=folder,
            df_angle_correction=df_angle_correction_empty,
            pad=0
        )

    print(f"Rectified images saved to: {out_dir}")
    print(f"Rectified rectangle size on final images: width={rect_w_px}px, height={rect_h_px}px")
    pixel_size = (75 + 56) / (rect_w_px + rect_h_px) # mm/pixel

    # Визуализируем результаты на видео
    print('Preparing video for visualization of image corrections...')

    out_video = make_comparison_video(
            main_folder=folder,
            in_subfolder="Spectral_Cube",
            out_subfolder="Spectral_Cube_Rectified",
            video_name="comparison",
            fps=10,
            hold_frames=2,
            wavelength_list=range(400, 1001, 10),
        )

    print(f"Saved video: {out_video}")

    # Детектируем объекты на скорректированных (выровненных друг с другом и выпрямленных изображениях)

    import os
    import pandas as pd

    wave_for_segmentation = '460'
    image_path_empty = folder + r'\Spectral_Cube_Rectified\image' +str(wave_for_segmentation) + '.jpg'

    if os.path.exists(image_path_empty):
        pass
    else:
        image_path_empty = folder + r'\Spectral_Cube_Rectified\image' +str(wave_for_segmentation) + '.png'

    detections_empty, image_empty = detect_n_objects(image_path_empty, 100)

    if empty:
        cls = 2
    else:
        cls = 0
    
    df_detections_empty = pd.DataFrame(detections_empty).reset_index()
    df_detections_empty.columns = ['cell', 'x_min', 'y_min', 'x_max', 'y_max', 'conf', 'cls']
    df_box_prompts_empty = df_detections_empty[['cell', 'x_min', 'y_min', 'x_max', 'y_max']][df_detections_empty['cls'] == cls]

    #plt.imshow(image_empty)
    #print(df_box_prompts_empty)

    # Последовательно используем боксы-подсказки для сегментации отдельных объектов
    n = len(df_box_prompts_empty)
    i = 0

    for cell in range(n):
        boxes = df_box_prompts_empty.iloc[cell][['x_min', 'y_min', 'x_max', 'y_max']].to_list()
        cell = int(df_box_prompts_empty.iloc[cell]['cell'])
        x1, y1, x2, y2 = boxes
        saved_mask_path = segment_with_sam_bbox(image_path_empty, x1, y1, x2, y2, cell, model_path="sam2.1_s.pt")
        print(f"\rMaking masks for each grain or empty cell: {i+1}/{n}", end="", flush=True)
        i += 1

    print("\nGrain masks done!")

    # Маски готовы. Теперь собираем геометрическую информацию о ячейках с их помощью.
    print('Gathering geometry data...')

    geometry_empty = []

    for cell in range(len(df_box_prompts_empty)):
        cell_number = int(df_box_prompts_empty.iloc[cell]['cell'])
        mask_path = folder + '/Spectral_Cube_Rectified/masks/' + 'image' + wave_for_segmentation + '_cell' + str(cell_number).zfill(3) + '.png'
        result = compute_mask_metrics(mask_path, threshold=127, pixel_size=pixel_size) # Задаем размер пикселя в мм
        geometry_empty.append([cell_number] + result)

    df_geometry_empty = pd.DataFrame(geometry_empty, columns=['cell', 'area[mm²]', 'perimeter[mm]', 'length[mm]', 'width[mm]'])

    # Геометрия ячеек готова, теперь перейдем к спектральным параметрам.
    print('Gathering spectral data...')

    df_spectra_empty = build_df_spectra_from_masks(folder, wavelength_list, df_box_prompts_empty)

    # Переводим стандартное отклонение в коэффицитент вариации

    df_spectra_empty = sd_to_cv(df_spectra_empty, wavelength_list=wavelength_list)

    df_all_data_empty = pd.concat([df_geometry_empty, df_spectra_empty[df_spectra_empty.columns[1:]]], axis=1)
    
    if empty:
        path = folder + '/empty_cell_data.xlsx'
    else:
        path = folder + '/grain_data.xlsx'
    
    df_all_data_empty.to_excel(path)
    
    return path

