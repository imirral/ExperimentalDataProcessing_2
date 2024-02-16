from classes.in_out import In_Out
from classes.processing import Processing

def main():
    in_out = In_Out()
    processing = Processing()

    file_name = 'grace/grace'
    if_color = False

    # original
    img = in_out.read_jpg(file_name)
    in_out.show_jpg(img, if_color)
    print("Размер изображения: " + str(img.shape))

    # shift
    plus_arr = processing.shift_2d(img, 30)
    in_out.write_jpg(plus_arr, file_name + '_shift')
    in_out.show_jpg(in_out.read_jpg(file_name + '_shift'), if_color)

    # multModel
    umn_arr = processing.multModel_2d(img, 1.3)
    in_out.write_jpg(umn_arr, file_name + '_mult')
    in_out.show_jpg(in_out.read_jpg(file_name + '_mult'), if_color)