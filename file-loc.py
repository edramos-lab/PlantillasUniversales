import os
import csv

# Specify the path you want to list the folder names from
path = r"C:/Users/edgar/OneDrive/Documentos/MLOPS/PlantillasUniversales/things-8/train/"

# Get the full paths of the folders in the specified path
folder_paths = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]

# Specify the output CSV file
output_csv = "folder_paths.csv"

# Write the full paths to the CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Folder Path"])  # Header
    for folder_path in folder_paths:
        writer.writerow([folder_path])

print(f"Folder paths have been written to {output_csv}")