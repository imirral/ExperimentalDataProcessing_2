from classes.in_out import In_Out
from classes.analysis import Analysis
from classes.processing import Processing
from classes.model import Model

in_out = In_Out()
processing = Processing()
model = Model()
analysis = Analysis()

is_color = False


def main():
    model_name = 'model'
    grace_name = 'grace'

    def plot(img_name, limit1, limit2, limit3, sigma=0.5, mask=3):
        # Оригинальное изображение
        img_data = in_out.read_jpg_file(img_name + '/' + img_name)

        # Зашумленное изображение
        random_noise = model.noise_2d(img_data.shape, 0.01)
        noisy_data = img_data + random_noise

        # Отфильтрованное изображение
        filtered_data = processing.average_filter(noisy_data, mask)

        in_out.show_jpg_files([img_data, noisy_data, filtered_data],
                              ['original', 'noised image',
                               f'average filter (mask {mask}x{mask})'],
                              is_color)

        def without_noise():
            hpf_image = processing.hpf_2d(img_data, sigma)
            lpf_image = processing.lpf_2d(img_data, sigma)

            hpf_threshold = processing.threshold(hpf_image, limit1)
            lpf_threshold = processing.threshold(img_data - lpf_image, limit1)

            in_out.show_jpg_files([img_data, hpf_threshold, lpf_threshold],
                                  ['original',
                                   f'HPF (sigma = {sigma}, limit = {limit1})',
                                   f'LPF (sigma = {sigma}, limit = {limit1})'],
                                  is_color)

        def with_noise():
            hpf_image = processing.hpf_2d(noisy_data, sigma)
            lpf_image = processing.lpf_2d(noisy_data, sigma)

            hpf_threshold = processing.threshold(hpf_image, limit2)
            lpf_threshold = processing.threshold(noisy_data - lpf_image, limit2)

            in_out.show_jpg_files([noisy_data, hpf_threshold, lpf_threshold],
                                  ['noisy image',
                                   f'HPF (sigma = {sigma}, limit = {limit2})',
                                   f'LPF (sigma = {sigma}, limit = {limit2})'],
                                  is_color)

        def filtered():
            hpf_image = processing.hpf_2d(filtered_data, sigma)
            lpf_image = processing.lpf_2d(filtered_data, sigma)

            hpf_threshold = processing.threshold(hpf_image, limit3)
            lpf_threshold = processing.threshold(filtered_data - lpf_image, limit3)

            in_out.show_jpg_files([filtered_data, hpf_threshold, lpf_threshold],
                                  [f'filtered image (mask {mask}x{mask})',
                                   f'HPF (sigma = {sigma}, limit = {limit3})',
                                   f'LPF (sigma = {sigma}, limit = {limit3})'],
                                  is_color)

        without_noise()
        with_noise()
        filtered()

    plot(model_name, 50, 10, 2)
    plot(grace_name, 6, 4, 1)
