from common import main_test as main
from option import args
import os

args.task = 'SSR'
args.angRes = 5
args.angRes_in_for_test = args.angRes
args.angRes_out_for_test = args.angRes

args.patch_size_for_test = 32
args.stride_for_test = 16
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.scale_factor = 4

    args.data_list_for_test = ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry']

    args.model_name = 'EPIT_SSR'
    args.path_pre_pth = '.pth/SSR/xx.pth'
    main(args)

