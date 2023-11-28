import os


def move_file_to_folder(folder_in, folder_out):
    for file in os.listdir(folder_in):
        if not os.path.isdir(os.path.join(folder_in, file)):
            os.rename(os.path.join(folder_in, file), os.path.join(folder_out, file))


def remove_xml_files(path):
    for file in os.listdir(path):
        if file.endswith(".xml"):
            os.remove(os.path.join(path, file))


def test(perfect, broken):
    files_perfect = [file for file in os.listdir(perfect)]
    files_broken = [file for file in os.listdir(broken)]

    dict_f = {}
    for f1 in files_perfect:
        for f2 in files_broken:
            if f2.startswith(f1.replace(".png", "")):
                if f1 in dict_f:
                    dict_f[f1].add(f2)
                else:
                    pass


"""
    for i in range(4):
        f1 = f"F:/PycharmProjects/dl-music-scores-restoration/dataset/pairs/broken/subset_{i}"
        f2 = "F:/PycharmProjects/dl-music-scores-restoration/dataset/pairs/broken"
        remove_xml_files(f1)
        move_file_to_folder(f1, f2)"""
if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())
