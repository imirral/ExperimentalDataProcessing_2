from classes.in_out import In_Out
from classes.processing import Processing

def main():
    in_out = In_Out()
    processing = Processing()

    file_name = 'c12-85v'
    shape = (1024, 1024)

    # c12-85v
    file_data = in_out.read_xcr_file(file_name, shape)
    file_data_recount = processing.recount_2d(file_data, 255)

    in_out.show_jpg_files([file_data_recount], [file_name], False)
    in_out.write_jpg_file(file_data_recount, 'c12-85v/' + file_name)

    in_out.write_bin_file(file_data_recount, 'x-ray_' + file_name)