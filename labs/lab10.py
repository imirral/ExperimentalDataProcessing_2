from classes.in_out import In_Out
from classes.processing import Processing


def main():
    in_out = In_Out()
    processing = Processing()

    file_name = 'grace'
    big_coef = 1.5
    small_coef = 1 / 1.5
    is_color = False

    img = in_out.read_jpg_file(file_name + '/' + file_name)

    big_img = processing.reshape_fourier_big(img, big_coef)

    small_img = processing.reshape_fourier_small(big_img, small_coef)

    difference = img - small_img

    in_out.show_jpg_files([big_img, small_img, difference],
                          ['grace_fourier_big ' + str(big_img.shape),
                           'grace_fourier_small ' + str(small_img.shape),
                           'grace - grace_fourier_small'],
                          is_color)
