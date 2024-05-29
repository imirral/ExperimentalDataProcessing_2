import numpy as np

from classes.model import Model
from classes.in_out import In_Out
from classes.analysis import Analysis
from classes.processing import Processing

model = Model()
in_out = In_Out()
analysis = Analysis()
processing = Processing()

is_color = False

mask_prewitt = [np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]]),
                np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]]),
                np.array([[0, 1, 1],
                          [-1, 0, 1],
                          [-1, -1, 0]]),
                np.array([[-1, -1, 0],
                          [-1, 0, 1],
                          [0, 1, 1]])]


def read_bin_file(file_name):
    data = np.fromfile(f'data/bin/{file_name}.bin', dtype="uint16")
    substr = file_name.split('x')
    size = int(substr[1])
    shape = (size, size)
    data = np.asarray(data).reshape(shape)
    return data


def data_enhancement(data):
    # Пересчет 2D данных
    step1 = model.recount_2d(data, 255)

    # Фильтрация с помощью градиентного фильтра (Превитт)
    step2 = processing.filter_with_gradient(step1, mask_prewitt[0], mask_prewitt[1])

    # Ослабление эффекта фильтрации
    step3 = np.asarray(step2 / 3, int)

    # Комбинирование оригинальных и обработанных данных
    step4 = model.recount_2d(step1 + step3, 255)

    # Градационное преобразование (эквализация гистограммы)
    step5 = processing.adjust_histogram_bounds(step4)

    in_out.show_jpg_files([step1, step2, step3, step4, step5],
                          ['step №1', 'step №2', 'step №3', 'step №4', 'step №5'],
                          is_color)

    in_out.show_jpg_files([step1, step5],
                          ['original', 'changed'],
                          is_color)

    return step5


def main():
    file_names = ['brain-H_x512', 'brain-V_x256', 'spine-H_x256', 'spine-V_x512']

    for file_name in file_names:
        data = read_bin_file(file_name)
        in_out.write_jpg_file(data, file_name + '/' + file_name)

        result = data_enhancement(data)
        in_out.write_jpg_file(result, file_name + '/' + file_name + '_changed')
