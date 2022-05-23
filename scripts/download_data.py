from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_file(url, destination_file):
    local_filename = destination_file
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return destination_file

def main():
    """
    Download data from the internet.
    """
    # Download data
    print("Creating folders")
    Path("runs/2022-03-03").mkdir(parents=True, exist_ok=True)
    Path("tmp/").mkdir(parents=True, exist_ok=True)

    # Download weights from following link
    print("Downloading weights...")
    link = "https://drive.google.com/u/0/uc?id=1i11elyNd6padoh8OVLGkOWGn2MxvRODy&export=download&confirm=t"
    weights = download_file(link, "tmp/weights.zip")
    # Unzip weights
    print("Unzipping weights...")
    with zipfile.ZipFile(weights, 'r') as z:
        z.extractall("runs/2022-03-03/12-33-21")
    
    # Delete __MACOSX and *.DS_Store files
    shutil.rmtree('runs/2022-03-03/12-33-21/__MACOSX', ignore_errors=True)
    for x in Path("runs/2022-03-03/12-33-21").glob('*/*.DS_Store'):
        x.unlink()

    print("Downloading Data")
    # Download test data
    link = "https://drive.google.com/u/0/uc?id=1ogers2BQQLPgZo3uC0NuT3xtwWgcMsGY&export=download"
    data = download_file(link, "tmp/sample_test_data.zip")

    # Unzip test data
    print("Unzipping test data...")
    with zipfile.ZipFile(data, 'r') as z:
        z.extractall("data/variable_currents")

    # Delete __MACOSX errors
    shutil.rmtree('data/variable_currents/__MACOSX', ignore_errors=True)

    print("Everything is done, deleting tmp folder")
    # Delete tmp folder and all its contents
    for x in Path("tmp/").glob("**/*"):
        x.unlink()
    Path("tmp/").rmdir()

if __name__ == "__main__":
    main()