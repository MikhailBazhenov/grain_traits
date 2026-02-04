# Собираем данные в один файл

def combine_data(main_folder, folder_list):

    import os
    import pandas as pd

    # Example inputs (assumed to already exist in your environment)
    # folder_list = ["folder1", "folder2", "folder3"]
    # wavelength_list = list(range(400, 1001, 10))

    all_dfs = []

    for folder in folder_list:
        file_path = os.path.join(main_folder, folder, "grain_data_corrected.xlsx")
        
        if not os.path.isfile(file_path):
            print(f"Warning: file not found -> {file_path}")
            continue
        
        # Read Excel, keeping the unnamed index column
        df = pd.read_excel(file_path, index_col=0)
        
        # Insert folder column as the first column
        df.insert(0, "folder", folder)
        
        all_dfs.append(df)

    # Combine all dataframes
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=0)
        combined_df.reset_index(drop=True, inplace=True)
        
        # Save to current directory
        output_file = os.path.join(main_folder, "grain_data_combined.xlsx")
        combined_df.to_excel(output_file)
        
        print(f"Saved combined data to: {output_file}")
    else:
        print("No data files were found. Output file was not created.")

if __name__ == '__main__':
    
    combine_data(main_folder=main_folder, folder_list=folder_list)

