from classes.in_out import In_Out
from classes.processing import Processing

def main():
    in_out = In_Out()
    processing = Processing()

    file_name = 'c12-85v'
    shape = (1024, 1024)

    file_data = in_out.read_xcr(file_name, shape)
    file_data_recount = processing.recount_2d(file_data, 255)

    in_out.show_jpg(file_data_recount, False)

    in_out.write_xcr(file_data_recount, 'x-ray_' + file_name)