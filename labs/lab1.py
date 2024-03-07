from classes.in_out import In_Out
from classes.model import Model

def main():
    in_out = In_Out()
    model = Model()

    file_name = 'grace'
    if_color = False

    # grace
    img = in_out.read_jpg_file('grace/' + file_name)

    # shift
    img_shift = model.shift_2d(img, 30)
    in_out.write_jpg_file(img_shift, 'grace/' + file_name + '_shift')
    img_shift = in_out.read_jpg_file('grace/' + file_name + '_shift')

    # multModel
    img_mult = model.mult_model_2d(img, 1.3)
    in_out.write_jpg_file(img_mult, 'grace/' + file_name + '_mult')
    img_mult = in_out.read_jpg_file('grace/' + file_name + '_mult')

    in_out.show_jpg_files([img, img_shift, img_mult],
                          ['grace ' + str(img.shape),
                           'grace_shift ' + str(img_shift.shape),
                           'grace_mult ' + str(img_mult.shape)],
                          if_color)