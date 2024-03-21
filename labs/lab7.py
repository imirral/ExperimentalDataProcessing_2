import matplotlib.pyplot as plt

from classes.in_out import In_Out
from classes.processing import Processing
from classes.model import Model

in_out = In_Out()
if_color = False


def main():
    processing = Processing()
    model = Model()

    img_name = 'model'
    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    def get_random_noisy_data(data):
        random_noise = model.noise_2d(data.shape)
        result = model.recount_2d(data + random_noise, 255)

        return result

    random_noisy_data = get_random_noisy_data(img_data)
    impulse_noisy_data = model.impulse_noise_2d(img_data)
    noisy_data = model.impulse_noise_2d(random_noisy_data)

    in_out.show_jpg_files([img_data, random_noisy_data, impulse_noisy_data, noisy_data],
                          ['original', 'random_noise', 'salt&pepper', 'random_noise + salt&pepper'],
                          if_color)

    def filter(mask):
        random_average_filter = processing.average_filter(random_noisy_data, mask)
        impulse_average_filter = processing.average_filter(impulse_noisy_data, mask)
        noisy_average_filter = processing.average_filter(noisy_data, mask)

        random_median_filter = processing.median_filter(random_noisy_data, mask)
        impulse_median_filter = processing.median_filter(impulse_noisy_data, mask)
        noisy_median_filter = processing.median_filter(noisy_data, mask)

        plt.figure()
        plt.suptitle(f'Mask {mask}x{mask} average/median filter', fontsize=15)

        plt.subplot(231)
        in_out.show_jpg_sub(random_average_filter, if_color, 'random_noise')
        plt.subplot(232)
        in_out.show_jpg_sub(impulse_average_filter, if_color, 'salt&pepper')
        plt.subplot(233)
        in_out.show_jpg_sub(noisy_average_filter, if_color, 'random_noise + salt&pepper')

        plt.subplot(234)
        in_out.show_jpg_sub(random_median_filter, if_color, 'random_noise')
        plt.subplot(235)
        in_out.show_jpg_sub(impulse_median_filter, if_color, 'salt&pepper')
        plt.subplot(236)
        in_out.show_jpg_sub(noisy_median_filter, if_color, 'random_noise + salt&pepper')

        plt.show()

    for msk in range(3, 11, 2):
        filter(msk)
