

def spectral_correction(folder_grain, folder_empty, wavelength_list):
    
    # Now we will make correction of the grain data using empty plate data
    import pandas as pd
    from filter_cells_by_global_mean_percentiles import filter_cells_by_global_mean_percentiles
    from scale_means_per_cell import scale_means_per_cell


    df_empty_plate = pd.read_excel(folder_empty + r'/empty_cell_data.xlsx', sheet_name='Sheet1', index_col=0)
    df_grain_plate = pd.read_excel(folder_grain + r'/grain_data.xlsx', sheet_name='Sheet1', index_col=0)

    # Делаем фильтрацию пересвеченных ячеек

    df_empty_plate = filter_cells_by_global_mean_percentiles(
        df_empty_plate,
        low_pct=20,
        high_pct=80,
        max_outside_pct=50
    )

    meta_cols = ['cell', 'area[mm²]', 'perimeter[mm]', 'length[mm]', 'width[mm]']
    mean_cols = [f"mean_{wl}" for wl in wavelength_list]
    all_cols   = df_grain_plate.columns.to_list()
    other_cols = [x for x in all_cols if x not in (meta_cols + mean_cols)]

    grain = df_grain_plate.set_index('cell')
    empty = df_empty_plate.set_index('cell')

    # Align empty to grain cells
    empty_aligned = empty.reindex(grain.index)

    # Optionally drop unmatched cells
    mask_matched = empty_aligned[mean_cols].notna().all(axis=1)
    grain = grain.loc[mask_matched]
    empty_aligned = empty_aligned.loc[mask_matched]

    # Compute corrected blocks
    meta_df = grain[meta_cols[1:]]  # without 'cell' (index)

    mean_corr = grain[mean_cols].to_numpy() / empty_aligned[mean_cols].to_numpy()

    mean_df = pd.DataFrame(mean_corr, index=grain.index, columns=mean_cols)
    other_df   = pd.DataFrame(grain, index=grain.index, columns=other_cols)

    # Concatenate everything at once (NO fragmentation)
    df_grain_corrected_plate = (
        pd.concat([meta_df, mean_df, other_df], axis=1)
        .reset_index()
        [df_grain_plate.columns]
    )

    df_grain_corrected_plate = scale_means_per_cell(
        df_grain_corrected_plate,
        wavelength_list=wavelength_list,
        low_wave=460,
        high_wave=1000
    )
    
    path = folder_grain + r'/grain_data_corrected.xlsx'
    df_grain_corrected_plate.to_excel(path)

    return path

