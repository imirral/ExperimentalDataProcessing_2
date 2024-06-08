from classes.model import Model
from classes.in_out import In_Out
from classes.analysis import Analysis
from classes.processing import Processing

model = Model()
in_out = In_Out()
analysis = Analysis()
processing = Processing()

is_color = False

c = 1
gamma = 0.4


def main():
    file_names = ['brain-H_x512', 'brain-V_x256', 'spine-H_x256', 'spine-V_x512']

    for file_name in file_names:
        # Считывание бинарного изображения
        # image_data = in_out.read_bin_file(file_name)
        # in_out.write_jpg_file(image_data, f'{file_name}/{file_name}')

        # Считывание изображения
        image_data = in_out.read_jpg_file(f'{file_name}/{file_name}')

        # Порогове преобразование (сегментация)
        step_1 = processing.threshold(image_data, 8)

        # Применение градационных преобразований к выделенному сегменту
        step_2 = apply_gamma_transform(image_data, step_1)
        step_3 = apply_histogram_equalization(step_2, step_1)

        in_out.write_jpg_file(step_3, f'{file_name}/{file_name}_changed')

        in_out.show_jpg_files([image_data, step_1, step_2, step_3],
                              ['original', 'segmented', 'gamma transform', 'histogram equalization'],
                              is_color)

        changed_data = in_out.read_jpg_file(f'{file_name}/{file_name}_changed')

        in_out.show_jpg_files([image_data, changed_data],
                              ['original', 'changed'],
                              is_color)


def apply_gamma_transform(image_data, mask):
    # Выделение данных обекта
    object_segment = image_data[mask == 255]

    # Гамма-коррекция данных объекта
    transformed_object_segment = processing.gamma_transform(object_segment, c, gamma)

    # Наложение преобразованных данных на исх. изображение
    transformed_image = image_data.copy()
    transformed_image[mask == 255] = transformed_object_segment

    return transformed_image


def apply_histogram_equalization(image_data, mask):
    # Выделение данных обекта
    object_segment = image_data[mask == 255]

    # Эквализация гистограммы данных объекта
    transformed_object_segment = processing.histogram_equalization(object_segment)

    # Наложение преобразованных данных на исх. изображение
    transformed_image = image_data.copy()
    transformed_image[mask == 255] = transformed_object_segment

    return transformed_image
