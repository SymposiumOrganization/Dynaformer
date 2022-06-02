from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import shutil
import click

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

@click.command()
@click.option("dataset_type", "--dataset_type", default="constant", help="Type of data to download")
def main(dataset_type):
    """
    Download the the training dataset
    """
    if dataset_type == "variable":
        link = "https://drive.google.com/u/0/uc?id=1lrJpDK5hOE-rQtM1wp3C6pEl1YOoTRAR&export=download&confirm=t"
    elif dataset_type == "constant":
        raise NotImplementedError
    else: 
        raise NotImplementedError

    # Download data
    print("Creating folders")
    Path("tmp/").mkdir(parents=True, exist_ok=True)
    Path("data/").mkdir(parents=True, exist_ok=True)

    # Download weights from following link
    print(f"Downloading {dataset_type} tradining dataset...")
    tmp_link = download_file(link, f"tmp/training_{dataset_type}.zip")

    # Unzip download
    print("Unzipping...")
    with zipfile.ZipFile(tmp_link, 'r') as z:
        z.extractall(f"tmp/{dataset_type}_currents")
    
    # Move from tmp to data 2022-02-19/17-07-11
    print("Moving...")
    shutil.move(f"tmp/{dataset_type}_currents/output_dataset_v2/random_variable_loading/2022-02-19", f"data/{dataset_type}_currents")
    
    # Delete tmp folder and all its contents
    shutil.rmtree("tmp/")


    # # Delete __MACOSX and *.DS_Store files
    # shutil.rmtree('runs/2022-03-03/12-33-21/__MACOSX', ignore_errors=True)
    # for x in Path("runs/2022-03-03/12-33-21").glob('*/*.DS_Store'):
    #     x.unlink()

    # print("Downloading Data")
    # # Download test data
    # link = "https://drive.google.com/u/0/uc?id=1ogers2BQQLPgZo3uC0NuT3xtwWgcMsGY&export=download"
    # data = download_file(link, "tmp/sample_test_data.zip")

    # # Unzip test data
    # print("Unzipping test data...")
    # with zipfile.ZipFile(data, 'r') as z:
    #     z.extractall("data/variable_currents")

    # # Delete __MACOSX errors
    # shutil.rmtree('data/variable_currents/__MACOSX', ignore_errors=True)

    # print("Everything is done, deleting tmp folder")
 
if __name__ == "__main__":
    main()