from classes.in_out import In_Out
from classes.processing import Processing
from classes.analysis import Analysis

def main():
    in_out = In_Out()
    analysis = Analysis()
    processing = Processing()

    if_color = False

    # Задание 1

    files = ('photo1', 'photo2', 'photo3', 'photo4', 'HollywoodLC')

    for file in files:
        data = in_out.read_jpg_file(file + '/' + file)
        hist = analysis.hist_2d(data)
        grad = processing.gradation_transform(data, hist)

        in_out.write_jpg_file(grad, file + '/' + file + '_gradation_transform')
        in_out.show_jpg_files([data, grad],
                              [file, file + '_gradation_transform'],
                              if_color)

    # Задание 2

    img_name = 'grace'
    small_coef = 1 / 1.2

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)
    big_img_neighbor = in_out.read_jpg_file(img_name + '/' + img_name + '_big_neighbor')
    big_img_interpol = in_out.read_jpg_file(img_name + '/' + img_name + '_big_interpol')

    # Уменьшение grace методом ближайшего соседа
    changed_img_neighbor = in_out.reshape_nearest_neighbor(big_img_neighbor, small_coef)

    # Уменьшение grace билинейной интерполяцией
    changed_img_interpol = in_out.reshape_bilinear_interpolation(big_img_interpol, small_coef)

    subtract_neighbor = img_data - changed_img_neighbor
    subtract_interpol = img_data - changed_img_interpol

    for y in range(img_data.shape[1]):
        for x in range(img_data.shape[0]):
            subtract_neighbor[x, y] += 1
            subtract_interpol[x, y] += 1

    in_out.write_jpg_file(subtract_neighbor, img_name + '/' + img_name + '_subtract_neighbor')
    in_out.write_jpg_file(subtract_interpol, img_name + '/' + img_name + '_subtract_interpol')

    in_out.show_jpg_files([img_data, changed_img_neighbor, subtract_neighbor],
                          [img_name, img_name + '_changed_neighbor',
                           img_name + '_subtraction_neighbor'],
                          if_color)

    in_out.show_jpg_files([img_data, changed_img_interpol, subtract_interpol],
                          [img_name, img_name + '_changed_interpol',
                           img_name + '_subtraction_interpol'],
                          if_color)