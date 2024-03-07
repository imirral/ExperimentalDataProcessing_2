import numpy as np

from classes.in_out import In_Out
from classes.processing import Processing

def main():
    in_out = In_Out()
    processing = Processing()

    big_coef = 1.2
    small_coef = 0.7
    screen_height = 256

    if_color = False

    # grace
    file_name = 'grace'

    img = in_out.read_jpg_file('grace/' + file_name)

    # Увеличение методом ближайшего соседа
    big_img_neighbor = in_out.reshape_nearest_neighbor(img, big_coef)
    in_out.write_jpg_file(big_img_neighbor, 'grace/' + file_name + '_big_neighbor')

    # Увеличение билинейной интерполяцией
    big_img_interpol = in_out.reshape_bilinear_interpolation(img, big_coef)
    in_out.write_jpg_file(big_img_interpol, 'grace/' + file_name + '_big_interpol')

    # Уменьшение методом ближайшего соседа
    small_img_neighbor = in_out.reshape_nearest_neighbor(img, small_coef)
    in_out.write_jpg_file(small_img_neighbor, 'grace/' + file_name + '_small_neighbor')

    # Уменьшение билинейной интерполяцией
    small_img_interpol = in_out.reshape_bilinear_interpolation(img, small_coef)
    in_out.write_jpg_file(small_img_interpol, 'grace/' + file_name + '_small_interpol')

    in_out.show_jpg_files([img, big_img_neighbor, big_img_interpol],
                          ['grace ' + str(img.shape),
                           'grace_big_neighbor ' + str(big_img_neighbor.shape),
                           'grace_big_interpol ' + str(big_img_interpol.shape)],
                          if_color)

    # c12-85v
    xcr_1_file_name = 'c12-85v'
    xcr_1_shape = (1024, 1024)

    xcr_1_data = in_out.read_xcr_file(xcr_1_file_name, xcr_1_shape)
    xcr_1_data_recount = np.rot90(processing.recount_2d(xcr_1_data, 255))

    xcr_1_coef = screen_height / xcr_1_data_recount.shape[0]

    # Уменьшение (высота изображения = высота экрана) методом ближайшего соседа
    xcr_1_resize_neighbor = in_out.reshape_nearest_neighbor(xcr_1_data_recount, xcr_1_coef)
    in_out.write_jpg_file(xcr_1_resize_neighbor, 'c12-85v/' + xcr_1_file_name + '_resize_neighbor')

    # Уменьшение (высота изображения = высота экрана) билинейной интерполяцией
    xcr_1_resize_interpol = in_out.reshape_bilinear_interpolation(xcr_1_data_recount, xcr_1_coef)
    in_out.write_jpg_file(xcr_1_resize_interpol, 'c12-85v/' + xcr_1_file_name + '_resize_interpol')

    in_out.show_jpg_files([xcr_1_data_recount, xcr_1_resize_neighbor, xcr_1_resize_interpol],
                          ['c12-85v ' + str(xcr_1_data_recount.shape),
                           'c12-85v_small_neighbor ' + str(xcr_1_resize_neighbor.shape),
                           'c12-85v_small_interpol ' + str(xcr_1_resize_interpol.shape)],
                          if_color)

    # u0
    xcr_2_file_name = 'u0'
    xcr_2_shape = (2500, 2048)

    xcr_2_data = in_out.read_xcr_file(xcr_2_file_name, xcr_2_shape)
    xcr_2_data_recount = np.rot90(processing.recount_2d(xcr_2_data, 255))

    xcr_2_coef = screen_height / xcr_2_data_recount.shape[0]

    # Уменьшение (высота изображения = высота экрана) методом ближайшего соседа
    xcr_2_resize_neighbor = in_out.reshape_nearest_neighbor(xcr_2_data_recount, xcr_2_coef)
    in_out.write_jpg_file(xcr_2_resize_neighbor, 'u0/' + xcr_2_file_name + '_resize_neighbor')

    # Уменьшение (высота изображения = высота экрана) билинейной интерполяцией
    xcr_2_resize_interpol = in_out.reshape_bilinear_interpolation(xcr_2_data_recount, xcr_2_coef)
    in_out.write_jpg_file(xcr_2_resize_interpol, 'u0/' + xcr_2_file_name + '_resize_interpol')

    in_out.show_jpg_files([xcr_2_data_recount, xcr_2_resize_neighbor, xcr_2_resize_interpol],
                          ['u0 ' + str(xcr_2_data_recount.shape),
                           'u0_small_neighbor ' + str(xcr_2_resize_neighbor.shape),
                           'u0_small_interpol ' + str(xcr_2_resize_interpol.shape)],
                          if_color)