from classes.in_out import In_Out
from classes.model import Model

def main():
    in_out = In_Out()
    model = Model()

    file_name = 'grace/grace'
    if_color = False

    # Оригинальные данные
    img = in_out.read_jpg_file(file_name)
    in_out.show_jpg_file(img, if_color, 'grace')
    print("Размер изображения: " + str(img.shape))

    # shift
    plus_arr = model.shift_2d(img, 30)
    in_out.write_jpg_file(plus_arr, file_name + '_shift')
    in_out.show_jpg_file(in_out.read_jpg_file(file_name + '_shift'), if_color, 'grace_shift')

    # multModel
    umn_arr = model.mult_model_2d(img, 1.3)
    in_out.write_jpg_file(umn_arr, file_name + '_mult')
    in_out.show_jpg_file(in_out.read_jpg_file(file_name + '_mult'), if_color, 'grace_mult')