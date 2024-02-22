from classes.in_out import In_Out
from classes.model import Model

def main():
    in_out = In_Out()
    model = Model()

    file_name = 'c12-85v'
    shape = (1024, 1024)

    # Данные с .xcr файла
    file_data = in_out.read_xcr_file(file_name, shape)
    file_data_recount = model.recount_2d(file_data, 255)

    in_out.show_jpg_file(file_data_recount, False, file_name)
    in_out.write_jpg_file(file_data_recount, 'c12-85v/' + file_name)

    in_out.write_bin_file(file_data_recount, 'x-ray_' + file_name)