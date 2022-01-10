from lib.image_matrix import VGGMatrix
from lib.utils import limit_gpu
from lib.clusterer import Clusterer


def main():
    limit_gpu()
    clusterer = Clusterer(image_dir_path = "./src_dir")
    clusterer.run()
    clusterer.save()



if __name__ == "__main__":
    main()
