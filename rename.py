import os


def main():
    # folder = "./dataset_new/test/co/open"
    # for filename in os.listdir(folder):
    #     dst = f"{str(1)+filename}"
    #     src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    #     dst = f"{folder}/{dst}"
    #
    #     os.rename(src, dst)

    folder = "./dataset_B_FacialImages/OpenFace"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"1_{str(count)}.jpg"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst}"

        os.rename(src, dst)


if __name__ == '__main__':

    main()
