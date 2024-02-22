import numpy as np
import soundfile as sf
import cv2 as cv2
import matplotlib.pyplot as plt
import struct

from scipy.io import wavfile

class In_Out:

    def read_dat_file(self, file_name):
        data = np.fromfile('data/dat/' + file_name, dtype="float32")
        return data

    def read_wav_file(self, file_name, rate):
        out_data = dict()
        samplerate, data = wavfile.read('data/wav/' + file_name)
        out_data['rate'] = samplerate
        out_data['data'] = data
        out_data['N'] = len(data)
        return out_data

    def write_wav_file(self, file_name, data, rate):
        sf.write('data/wav/' + file_name + '.wav', data, rate)

    def read_jpg_file(self, file_name):
        img = cv2.imread('data/jpg/' + file_name + '.jpg', cv2.IMREAD_GRAYSCALE)
        return img

    def show_jpg_file(self, img, if_color, name):
        if if_color:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')

        plt.title(name, fontsize=12)
        plt.axis('off')
        plt.autoscale(tight=True)

        plt.show()

    def write_jpg_file(self, array, file_name):
        cv2.imwrite('data/jpg/' + file_name + '.jpg', array)

    def read_xcr_file(self, file_name, shape):
        with open('data/xcr/' + file_name + '.xcr', 'rb') as file:
            header = file.read(2048)
            data = file.read(shape[0] * shape[1] * 2)  # Считываем нужное количество байт данных

            # Преобразование данных из двоичного формата
            image_data = np.array(struct.unpack('<' + 'H' * (shape[0] * shape[1]), data)).reshape(shape)

            # Перестановка байтов в каждом двухбайтовом числе (младший <-> старший байт)
            image_data.byteswap(True)

            return image_data

    def write_bin_file(self, data, file_name):
        with open('data/bin/' + file_name + '.bin', 'wb') as file:
            # Преобразование массива в бинарные данные
            binary_data = struct.pack('<' + 'H' * data.size, *data.flatten())

            file.write(binary_data)

    def read_bin_file(self, file_name, shape):
        with open('data/bin/' + file_name + '.bin', 'rb') as file:

            # Чтение необходимого количества байт данных
            binary_data = file.read(shape[0] * shape[1] * 2)

            data = np.array(struct.unpack('<' + 'H' * (shape[0] * shape[1]), binary_data)).reshape(shape)

            return data

    def reshape_nearest_neighbor(self, img, coef):
        # Определение новых размеров изображения
        new_width = int(img.shape[1] * coef)
        new_height = int(img.shape[0] * coef)

        # Изменение размера изображения с помощью метода ближайшего соседа
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        return resized_img

    def reshape_bilinear_interpolation(self, img, coef):
        # Определение новых размеров изображения
        new_width = int(img.shape[1] * coef)
        new_height = int(img.shape[0] * coef)

        # Изменение размера изображения с использованием билинейной интерполяции
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        return resized_img