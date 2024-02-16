import numpy as np
from scipy.io import wavfile
import soundfile as sf
import cv2 as cv2
import matplotlib.pyplot as plt
import struct

class In_Out:

    def read_dat(self, file_name):
        data = np.fromfile('data/dat/' + file_name, dtype="float32")
        return data

    def read_wav(self, file_name, rate):
        out_data = dict()
        samplerate, data = wavfile.read('data/wav/' + file_name)
        out_data['rate'] = samplerate
        out_data['data'] = data
        out_data['N'] = len(data)
        return out_data

    def write_wav(self, file_name, data, rate):
        sf.write('data/wav/' + file_name + '.wav', data, rate)

    def read_jpg(self, file_name):
        img = cv2.imread('data/jpg/' + file_name + '.jpg', cv2.IMREAD_GRAYSCALE)
        return img

    def show_jpg(self, img, if_color):
        if if_color:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.autoscale(tight=True)
        plt.show()

    def write_jpg(self, array, file_name):
        cv2.imwrite('data/jpg/' + file_name + '.jpg', array)

    def read_xcr(self, file_name, shape):
        with open('data/xcr/' + file_name + '.xcr', 'rb') as file:
            header = file.read(2048)
            data = file.read(shape[0] * shape[1] * 2)

            # Преобразование данных из двоичного формата
            image_data = np.array(struct.unpack('<' + 'H' * (1024 * 1024), data)).reshape((1024, 1024))

            # Перестановка байтов в каждом двухбайтовом числе (младший <-> старший байт)
            image_data.byteswap(True)

            return image_data

    def write_xcr(self, data, file_name):
        with open('data/bin/' + file_name + '.bin', 'wb') as file:
            # Преобразование массива в бинарные данные
            binary_data = struct.pack('<' + 'H' * data.size, *data.flatten())

            file.write(binary_data)