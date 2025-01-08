from roboflow import Roboflow
import sys

def download_dataset(project_version):
    rf = Roboflow(api_key="wv1RWb130ECTfhxSNxPS")
    project = rf.workspace("footdiseaseimgclass").project("things-jam67")
    version = project.version(project_version)
    dataset = version.download("folder")
    return dataset

def main(project_version):
    dataset = download_dataset(project_version)
    print("Dataset downloaded successfully")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python dataset.py <ProjectVersion>")
        sys.exit(1)
    project_version = int(sys.argv[1])
    main(project_version)