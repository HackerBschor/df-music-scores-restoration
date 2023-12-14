import os
from typing import Tuple, List, Set, Dict

import pandas as pd

# Filename endings of the DocCreator (https://doc-creator.labri.fr/) degenerations
doccreator_endings = ["Bleed_0", "Blur_Complete_0", "CharDeg_0", "Hole_0", "Phantom_FREQUENT_0", "Phantom_RARE_0",
                      "Shadow_Bottom", "Shadow_Left", "Shadow_Right", "Shadow_Top"]


def get_files_clean_dirty(folder_perfect: str, folder_broken: str) -> Tuple[Set[str], Set[str]]:
    files_clean: List[str] = [file for file in os.listdir(folder_perfect)]
    files_dirty: List[str] = [file for file in os.listdir(folder_broken)]
    return set(files_clean), set(files_dirty)


def create_dict_clean_dirty(files_clean: Set[str], files_dirty: Set[str]) -> Dict[str, list]:
    """
    Create a dictionary with the clean files as keys and a list of the according dirty files (created by DocCreator)
    """
    files_dict: Dict[str, list] = {}

    for i, f1 in enumerate(files_clean):
        print("\r", round(float(i) * 100 / float(len(files_clean)), 2), "%", end="")

        files_dict[f1] = []
        for ending in doccreator_endings:
            if f1.replace(".png", "")+ending+".png" in files_dirty:
                files_dict[f1].append(f1.replace(".png", "")+ending+".png")

    return files_dict


def convert_dict_clean_dirty_to_df(dict_clean_dirty):
    result_list = []

    # Iterate through the dictionary items
    for key, values in dict_clean_dirty.items():
        if len(values) > 0:
            result_list.extend([(key, value) for value in values])
        else:
            result_list.append((key, None))

    # Create a DataFrame from the list of tuples
    return pd.DataFrame(result_list, columns=['file_perfect', 'file_broken']).set_index("file_perfect")
