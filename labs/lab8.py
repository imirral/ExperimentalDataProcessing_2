import matplotlib.pyplot as plt

from classes.model import Model
from classes.analysis import Analysis
from classes.processing import Processing
from classes.in_out import In_Out


def plot_graph(item, data, name, fourier=False, xn=[], dt=0.002):
    plt.subplot(item)
    plt.title(name)

    if fourier:
        plt.plot(xn, data)
        plt.xlim([0, 1 / (dt * 2)])
    else:
        plt.plot(data)


def main():
    model = Model()
    analysis = Analysis()
    in_out = In_Out()
    processing = Processing()

    def task1():
        n = 10 ** 3
        dt = 0.002

        harm = model.harm(n, 100, 33, 0.001)

        harm_fourier = analysis.fourier(harm)
        harm_inverse = analysis.inverse_fourier(harm_fourier)

        new_x_n = analysis.spectrum_fourier([i for i in range(n)], n, dt)

        plt.figure()
        plot_graph(311, harm, 'data')
        plot_graph(312, harm_fourier, 'fourier', True, new_x_n)
        plot_graph(313, harm_inverse, 'inverse_fourier')
        plt.show()

    def task2():
        img_name = 'rect'

        # test_img = model.get_rectangle((256, 256), (20, 30))
        # in_out.write_jpg_file(test_img, img_name + '/' + img_name)

        test_img = in_out.read_jpg_file(img_name + '/' + img_name)

        spectrum = analysis.fourier_2D(test_img)
        centered_spectrum = analysis.fourier_rearrange(spectrum)

        in_out.show_jpg_files([test_img, spectrum, centered_spectrum],
                              ['original', 'spectrum_fourier', 'centered_spectrum'],
                              False)

    def task3():
        img_name = 'rect'

        test_img = in_out.read_jpg_file(img_name + '/' + img_name)

        spectrum = analysis.fourier_2D(test_img)
        centered_spectrum = analysis.fourier_rearrange(spectrum)

        inverse = analysis.inverse_fourier_2D(spectrum)

        in_out.show_jpg_files([test_img, centered_spectrum, inverse],
                              ['original', 'centered_spectrum', 'inverse_fourier'],
                              False)

    def task4():
        img_name = 'grace'

        img = in_out.read_jpg_file(img_name + '/' + img_name)

        spectrum = analysis.fourier_2D(img)
        centered_spectrum = analysis.fourier_rearrange(spectrum)

        in_out.write_jpg_file(centered_spectrum, f'{img_name}/{img_name}_spectrum')

        inverse = analysis.inverse_fourier_2D(spectrum)

        in_out.show_jpg_files([img, centered_spectrum, inverse],
                              ['original', 'centered_spectrum', 'inverse_fourier'],
                              False)

    # task1()
    # task2()
    # task3()
    # task4()
