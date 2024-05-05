import numpy as np

from classes.in_out import In_Out
from classes.processing import Processing
from classes.model import Model


in_out = In_Out()
processing = Processing()
model = Model()

is_color = False


def plot_threshold(img_data, limit, name):
    threshold = processing.threshold(img_data, limit)

    in_out.show_jpg_files([img_data, threshold],
                          [name, 'threshold'],
                          is_color)

    return threshold


def plot_morpho(img_data, size=3):
    mask = np.ones((size, size), np.uint8)

    erosion = processing.erosion(img_data, mask)
    dilation = processing.dilation(img_data, mask)

    sub_dilation = dilation - img_data
    sub_erosion = img_data - erosion

    in_out.show_jpg_files([erosion, dilation, sub_erosion, sub_dilation],
                          [f'erosion (mask {size}x{size})',
                           f'dilation (mask {size}x{size})',
                           f'dilation – threshold',
                           f'threshold – erosion'],
                          is_color)


def main():
    model_name = 'model'
    grace_name = 'grace'

    mask = 3

    def plot(img_name):
        # Оригинальное изображение
        img_data = in_out.read_jpg_file(img_name + '/' + img_name)

        # Зашумленное изображение
        random_noise = model.noise_2d(img_data.shape, 0.01)
        noisy_data = img_data + random_noise

        # Отфильтрованное изображение
        filtered_data = processing.average_filter(noisy_data, mask)
        filtered_data = np.array(filtered_data)

        in_out.show_jpg_files([img_data, noisy_data, filtered_data],
                              ['original', 'noised image',
                               f'average filter (mask {mask}x{mask})'],
                              is_color)

        def without_noise():
            threshold = plot_threshold(img_data, 200, 'original')
            plot_morpho(threshold)

        def with_noise():
            threshold = plot_threshold(noisy_data, 200, 'noised image')
            plot_morpho(threshold)

        def filtered():
            threshold = plot_threshold(filtered_data, 200, f'average filter (mask {mask}x{mask})')
            plot_morpho(threshold)

        without_noise()
        with_noise()
        filtered()

    plot(model_name)
    plot(grace_name)

