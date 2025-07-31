from common import main
from option import args

args.task = 'ASR'
args.angRes_in = 2
args.angRes_out = 7
args.scale_factor = 1

args.epoch = 80
args.batch_size = 4
args.lr = 2e-4
args.patch_for_train = 64

args.patch_size_for_test = 128
args.stride_for_test = 64
args.minibatch_for_test = 1

if __name__ == '__main__':
    args.device = 'cuda:0'
    args.data_list_for_train = ['RE_Lytro']
    args.data_list_for_test = ['RE_Lytro_reflective']

    args.model_name = 'EPIT_ASR'
    main(args)
