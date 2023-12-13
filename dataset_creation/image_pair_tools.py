import os
import pandas as pd


def get_file_lists(folder_perfect, folder_broken):
    files_perfect = [file for file in os.listdir(folder_perfect)]
    files_broken = [file for file in os.listdir(folder_broken)]
    return set(files_perfect), set(files_broken)


def create_perfect_broken_dict(files_perfect, files_broken):
    endings = ["Bleed_0", "Blur_Complete_0", "CharDeg_0", "Hole_0", "Phantom_FREQUENT_0", "Phantom_RARE_0", "Shadow_Bottom", "Shadow_Left", "Shadow_Right", "Shadow_Top"]

    files_dict = {}

    for i, f1 in enumerate(files_perfect):
        print("\r", round(float(i)*100/float(len(files_perfect)), 2), "%", end="")
        files_dict[f1] = []
        for ending in endings:
            if f1.replace(".png", "")+ending+".png" in files_broken:
                files_dict[f1].append(f1.replace(".png", "")+ending+".png")

    return files_dict


def dict_to_df(d):
    result_list = []

    # Iterate through the dictionary items
    for key, values in d.items():
        if len(values) > 0:
            # Append tuples to the result list
            result_list.extend([(key, value) for value in values])
        else:
            result_list.append((key, None))

    # Create a DataFrame from the list of tuples
    return pd.DataFrame(result_list, columns=['file_perfect', 'file_broken']).set_index("file_perfect")


if __name__ == '__main__':
    perfect = "F:/PycharmProjects/dl-music-scores-restoration/dataset/pairs/perfect"
    broken = "F:/PycharmProjects/dl-music-scores-restoration/dataset/pairs/broken"
    perfect, broken = get_file_lists(perfect, broken)
    dict_perfect_broken = create_perfect_broken_dict(perfect, broken)
    df = dict_to_df(dict_perfect_broken)
    print(df)
    df.to_pickle("../dataset/pairs/perfect_broken_index.pkl")
