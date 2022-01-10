from lib.image_matrix import VGGMatrix
from lib.utils import limit_gpu
from lib.utils import euclidean_dist


def main():
    limit_gpu()
    matrix_calculator = VGGMatrix()
    result = matrix_calculator.get_matrix(image ="/home/musa/Downloads/bird.jpg" )
    result2 = matrix_calculator.get_matrix(image ="/home/musa/Downloads/musa.jpg" )

    print(euclidean_dist(result, result))
    print(euclidean_dist(result, result2))



if __name__ == "__main__":
    main()
