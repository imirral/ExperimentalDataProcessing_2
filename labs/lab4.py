import numpy as np

from classes.in_out import In_Out
from classes.processing import Processing

def main():
    in_out = In_Out()
    processing = Processing()

    if_color = False

    # grace
    grace_file_name = 'grace'

    img_grace = in_out.read_jpg_file('grace/' + grace_file_name)
    neg_grace = processing.negative(img_grace, 255)

    in_out.write_jpg_file(neg_grace, 'grace/' + grace_file_name + '_negative')

    in_out.show_jpg_files([img_grace, neg_grace],
                          ['grace', 'grace_negative'],
                          if_color)

    # c12-85v
    xcr_1_file_name = 'c12-85v'
    xcr_1_shape = (1024, 1024)

    xcr_1_data = in_out.read_xcr_file(xcr_1_file_name, xcr_1_shape)
    xcr_1_data_recount = np.rot90(processing.recount_2d(xcr_1_data, 255))
    neg_xcr_1 = processing.negative(xcr_1_data_recount, 255)

    in_out.write_jpg_file(neg_xcr_1, 'c12-85v/' + xcr_1_file_name + '_negative')

    in_out.show_jpg_files([xcr_1_data_recount, neg_xcr_1],
                          ['c12-85v', 'c12-85v_negative'],
                          if_color)

    # u0
    xcr_2_file_name = 'u0'
    xcr_2_shape = (2500, 2048)

    xcr_2_data = in_out.read_xcr_file(xcr_2_file_name, xcr_2_shape)
    xcr_2_data_recount = np.rot90(processing.recount_2d(xcr_2_data, 255))
    neg_xcr_2 = processing.negative(xcr_2_data_recount, 255)

    in_out.write_jpg_file(neg_xcr_2, 'u0/' + xcr_2_file_name + '_negative')

    in_out.show_jpg_files([xcr_2_data_recount, neg_xcr_2],
                          ['u0', 'u0_negative'],
                          if_color)

    # photo1
    img_name = 'photo1'
    c_gamma = 1
    gamma = 0.4
    c_log = 35

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    img_gamma = processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = processing.log_transform(img_data, c_log)

    in_out.write_jpg_file(img_gamma, img_name + '/' + img_name + '_gamma')
    in_out.write_jpg_file(img_log, img_name + '/' + img_name + '_log')

    print('gamma_transform, gamma = ' + str(gamma) + ', C = ' + str(c_gamma))
    print('log_transform, C = ' + str(c_log))

    in_out.show_jpg_files([img_data, img_gamma, img_log],
                          [img_name, img_name + '_gamma', img_name + '_log'],
                          if_color)

    # photo2
    img_name = 'photo2'
    c_gamma = 1
    gamma = 0.4
    c_log = 30

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    img_gamma = processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = processing.log_transform(img_data, c_log)

    in_out.write_jpg_file(img_gamma, img_name + '/' + img_name + '_gamma')
    in_out.write_jpg_file(img_log, img_name + '/' + img_name + '_log')

    print('gamma_transform, gamma = ' + str(gamma) + ', C = ' + str(c_gamma))
    print('log_transform, C = ' + str(c_log))

    in_out.show_jpg_files([img_data, img_gamma, img_log],
                          [img_name, img_name + '_gamma', img_name + '_log'],
                          if_color)

    # photo3
    img_name = 'photo3'
    c_gamma = 1
    gamma = 0.4
    c_log = 40

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    img_gamma = processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = processing.log_transform(img_data, c_log)

    in_out.write_jpg_file(img_gamma, img_name + '/' + img_name + '_gamma')
    in_out.write_jpg_file(img_log, img_name + '/' + img_name + '_log')

    print('gamma_transform, gamma = ' + str(gamma) + ', C = ' + str(c_gamma))
    print('log_transform, C = ' + str(c_log))

    in_out.show_jpg_files([img_data, img_gamma, img_log],
                          [img_name, img_name + '_gamma', img_name + '_log'],
                          if_color)

    # photo4
    img_name = 'photo4'
    c_gamma = 1
    gamma = 0.67
    c_log = 20

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    img_gamma = processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = processing.log_transform(img_data, c_log)

    in_out.write_jpg_file(img_gamma, img_name + '/' + img_name + '_gamma')
    in_out.write_jpg_file(img_log, img_name + '/' + img_name + '_log')

    print('gamma_transform, gamma = ' + str(gamma) + ', C = ' + str(c_gamma))
    print('log_transform, C = ' + str(c_log))

    in_out.show_jpg_files([img_data, img_gamma, img_log],
                          [img_name, img_name + '_gamma', img_name + '_log'],
                          if_color)

    # HollywoodLC
    img_name = 'HollywoodLC'
    c_gamma = 1
    gamma = 0.67
    c_log = 30

    img_data = in_out.read_jpg_file(img_name + '/' + img_name)

    img_gamma = processing.gamma_transform(img_data, c_gamma, gamma)
    img_log = processing.log_transform(img_data, c_log)

    in_out.write_jpg_file(img_gamma, img_name + '/' + img_name + '_gamma')
    in_out.write_jpg_file(img_log, img_name + '/' + img_name + '_log')

    print('gamma_transform, gamma = ' + str(gamma) + ', C = ' + str(c_gamma))
    print('log_transform, C = ' + str(c_log))

    in_out.show_jpg_files([img_data, img_gamma, img_log],
                          [img_name, img_name + '_gamma', img_name + '_log'],
                          if_color)