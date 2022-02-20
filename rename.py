import os


def main():
    folder = "./data_mix/valid/"
    for filename in os.listdir(folder):
        if filename[0] == "0":
            dst = f"{str(2)+filename[1:]}"
        else:
            dst = f"{str(3)+filename[1:]}"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst}"

        os.rename(src, dst)

    # folder = "./dataset_B_FacialImages/OpenFace"
    # for count, filename in enumerate(os.listdir(folder)):
    #     dst = f"1_{str(count)}.jpg"
    #     src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    #     dst = f"{folder}/{dst}"
    #
    #     os.rename(src, dst)


if __name__ == '__main__':

    main()
