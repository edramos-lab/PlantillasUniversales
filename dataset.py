from roboflow import Roboflow

def download_dataset():
    rf = Roboflow(api_key="wv1RWb130ECTfhxSNxPS")
    project = rf.workspace("footdiseaseimgclass").project("things-jam67")
    version = project.version(1)
    dataset = version.download("folder")
    return dataset

def main():
    dataset = download_dataset()
    print("Dataset downloaded successfully")

if __name__ == "__main__":
    main()