import numpy as np

from classes.in_out import In_Out
from classes.model import Model

def main():
    in_out = In_Out()
    model = Model()

    file_name = 'grace/grace'
    big_coef = 1.3
    small_coef = 0.7
    xcr_1_file_name = 'c12-85v'
    xcr_1_shape = (1024, 1024)
    xcr_2_file_name = 'u0'
    xcr_2_shape = (2500, 2048)
    screen_height = 256
    if_color = False

    # Оригинальные данные
    img = in_out.read_jpg_file(file_name)

    xcr_1_data = in_out.read_xcr_file(xcr_1_file_name, xcr_1_shape)
    xcr_1_data_recount = np.rot90(model.recount_2d(xcr_1_data, 255))

    xcr_2_data = in_out.read_xcr_file(xcr_2_file_name, xcr_2_shape)
    xcr_2_data_recount = np.rot90(model.recount_2d(xcr_2_data, 255))

    # Увеличение grace методом ближайшего соседа
    bid_img_neighbor = in_out.reshape_nearest_neighbor(img, big_coef)
    in_out.write_jpg_file(bid_img_neighbor, file_name + '_big_neighbor')

    print("Размер изображения: " + str(bid_img_neighbor.shape))
    in_out.show_jpg_file(in_out.read_jpg_file(file_name + '_big_neighbor'), if_color, 'grace_big_neighbor')

    # Увеличение grace билинейной интерполяцией
    bid_img_interpol = in_out.reshape_bilinear_interpolation(img, big_coef)
    in_out.write_jpg_file(bid_img_interpol, file_name + '_big_interpol')

    print("Размер изображения: " + str(bid_img_interpol.shape))
    in_out.show_jpg_file(in_out.read_jpg_file(file_name + '_big_interpol'), if_color, 'grace_big_interpol')

    # Уменьшение grace методом ближайшего соседа
    small_img_neighbor = in_out.reshape_nearest_neighbor(img, small_coef)
    in_out.write_jpg_file(small_img_neighbor, file_name + '_small_neighbor')

    print("Размер изображения: " + str(small_img_neighbor.shape))
    in_out.show_jpg_file(in_out.read_jpg_file(file_name + '_small_neighbor'), if_color, 'grace_small_neighbor')

    # Уменьшение grace билинейной интерполяцией
    small_img_interpol = in_out.reshape_bilinear_interpolation(img, small_coef)
    in_out.write_jpg_file(small_img_interpol, file_name + '_small_interpol')

    print("Размер изображения: " + str(small_img_interpol.shape))
    in_out.show_jpg_file(in_out.read_jpg_file(file_name + '_small_interpol'), if_color, 'grace_small_interpol')

    xcr_1_coef = screen_height / xcr_1_data_recount.shape[0]

    # Изменение размера c12-85v (высота изображения = высота экрана) методом ближайшего соседа
    xcr_1_resize_neighbor = in_out.reshape_nearest_neighbor(xcr_1_data_recount, xcr_1_coef)
    in_out.write_jpg_file(xcr_1_resize_neighbor, 'c12-85v/' + xcr_1_file_name + '_resize_neighbor')

    print("Размер изображения: " + str(xcr_1_resize_neighbor.shape))
    in_out.show_jpg_file(in_out.read_jpg_file('c12-85v/' + xcr_1_file_name + '_resize_neighbor'), if_color, 'c12-85v_resize_neighbor')

    # Изменение размера c12-85v (высота изображения = высота экрана) билинейной интерполяцией
    xcr_1_resize_interpol = in_out.reshape_bilinear_interpolation(xcr_1_data_recount, xcr_1_coef)
    in_out.write_jpg_file(xcr_1_resize_interpol, 'c12-85v/' + xcr_1_file_name + '_resize_interpol')

    print("Размер изображения: " + str(xcr_1_resize_interpol.shape))
    in_out.show_jpg_file(in_out.read_jpg_file('c12-85v/' + xcr_1_file_name + '_resize_interpol'), if_color,
                         'c12-85v_resize_interpol')

    xcr_2_coef = screen_height / xcr_2_data_recount.shape[0]

    # Изменение размера u0 (высота изображения = высота экрана) методом ближайшего соседа
    xcr_2_resize_neighbor = in_out.reshape_nearest_neighbor(xcr_2_data_recount, xcr_2_coef)
    in_out.write_jpg_file(xcr_2_resize_neighbor, 'u0/' + xcr_2_file_name + '_resize_neighbor')

    print("Размер изображения: " + str(xcr_2_resize_neighbor.shape))
    in_out.show_jpg_file(in_out.read_jpg_file('u0/' + xcr_2_file_name + '_resize_neighbor'), if_color,
                         'u0_resize_neighbor')

    # Изменение размера u0 (высота изображения = высота экрана) билинейной интерполяцией
    xcr_2_resize_interpol = in_out.reshape_bilinear_interpolation(xcr_2_data_recount, xcr_2_coef)
    in_out.write_jpg_file(xcr_2_resize_interpol, 'u0/' + xcr_2_file_name + '_resize_interpol')

    print("Размер изображения: " + str(xcr_2_resize_interpol.shape))
    in_out.show_jpg_file(in_out.read_jpg_file('u0/' + xcr_2_file_name + '_resize_interpol'), if_color,
                         'u0_resize_interpol')