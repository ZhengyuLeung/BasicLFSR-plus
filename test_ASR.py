from common import main_test as main
from option import args


args.task = 'ASR'
args.angRes_in = 2
args.angRes_out = 7
args.scale_factor = 1

args.patch_size_for_test = 128
args.stride_for_test = 64
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.angRes_in_for_test = 2
    args.angRes_out_for_test = 7

    args.data_list_for_train = ['RE_HCI']
    args.data_list_for_test = ['RE_HCI_new', 'RE_HCI_old']
    
    
    args.model_name = 'Yeung2018'
    args.path_pre_pth = './pth/ASR/Yeung2018_HCI_2x2_7x7_ASR_epoch_80_model.pth'
    main(args)

    #######################################################
    args.data_list_for_train = ['RE_Lytro']
    args.data_list_for_test = ['RE_Lytro_30scene', 'RE_Lytro_occlusions', 'RE_Lytro_reflective']

    args.model_name = 'Yeung2018'
    args.path_pre_pth = './pth/ASR/Yeung2018_Lytro_2x2_7x7_ASR_epoch_80_model.pth'
    main(args)