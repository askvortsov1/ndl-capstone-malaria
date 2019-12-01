import io
import os
import random
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

from tqdm import tqdm

MALARIA_DATASET_URL = "ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip"

DATA_PATH = "data/malaria/"

DOWNLOAD_DATA = True

REMOVE_OLD_DATA = True

VAL_PCT = 0.1
TEST_PCT = 0.2
TRAIN_PCT = 1 - VAL_PCT - TEST_PCT

CLASSES = ["Parasitized", "Uninfected"]
DATASETS = ["train", "val", "test"]


def download_and_split_data():
    if REMOVE_OLD_DATA:
        shutil.rmtree(DATA_PATH, ignore_errors=True)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    for subdir in DATASETS:
        if not os.path.exists("{}{}/".format(DATA_PATH, subdir)):
            os.makedirs("{}{}/".format(DATA_PATH, subdir))
        for subdir2 in CLASSES:
            if not os.path.exists("{}{}/{}/".format(DATA_PATH, subdir, subdir2)):
                os.makedirs("{}{}/{}/".format(DATA_PATH, subdir, subdir2))

    if DOWNLOAD_DATA:
        data = urlopen(MALARIA_DATASET_URL)
        memfile = io.BytesIO(data.read())
        with ZipFile(memfile, 'r') as archive:
            archive.extractall()

    raw_image_sets = {x: os.listdir("cell_images/{}/".format(x)) for x in CLASSES}
    for i in raw_image_sets:
        random.shuffle(raw_image_sets[i])
        raw_image_sets[i] = [img for img in raw_image_sets[i] if "png" in img]

    total_files = min([len(raw_image_sets[i]) for i in raw_image_sets])

    num_val = int(VAL_PCT * total_files)
    num_test = int(TEST_PCT * total_files)

    image_sets = {}

    image_sets["val"] = {k: v[:num_val] for k, v in raw_image_sets.items()}
    image_sets["test"] = {k: v[-num_test:] for k, v in raw_image_sets.items()}
    image_sets["train"] = {k: v[num_val:-num_test] for k, v in raw_image_sets.items()}

    for dataset in DATASETS:
        print("Dataset: ", dataset)
        for dataclass in CLASSES:
            print("Class: ", dataclass)
            for i, path in tqdm(enumerate(image_sets[dataset][dataclass])):
                shutil.move("cell_images/{}/{}".format(dataclass, path), "{}/{}/{}/{}.png".format(DATA_PATH, dataset, dataclass, i))
            print()
        print()

    if DOWNLOAD_DATA:
        shutil.rmtree("cell_images", ignore_errors=True)
