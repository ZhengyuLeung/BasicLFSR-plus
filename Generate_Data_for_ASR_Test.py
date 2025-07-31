import argparse
import os
import h5py
import numpy as np
from pathlib import Path
import scipy.io as scio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ASR', help='SSR, ASR')
    parser.add_argument("--max_angRes", type=int, default=7, help="angular resolution")
    parser.add_argument('--data_for', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default='../datasets/', help='')
    parser.add_argument('--save_data_path', type=str, default='./', help='')

    return parser.parse_args()


def main(args):
    angRes = args.max_angRes
    ''' dir '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath(args.task + '_' + str(angRes) + 'x' + str(angRes))
    save_dir.mkdir(exist_ok=True)

    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['RE_HCI_new', 'RE_HCI_old', 'RE_Lytro_30scene', 'RE_Lytro_occlusions',
                                               'RE_Lytro_reflective']:
            continue
        name_dataset = src_datasets[index_dataset]
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)

        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        for root, _, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating test data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    # 加载光场数据
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(root + file)
                    LF = np.array(data['LF'])

                # 加载光场数据
                (U, V, H, W, _) = LF.shape
                # Extract central angRes_out * angRes_out views
                LF = LF[(U - angRes) // 2:(U + angRes) // 2, (V - angRes) // 2:(V + angRes) // 2, :, :, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape

                # 测试集不对 光场数据 切割 patches
                # idx_save = idx_save + 1
                idx_scene_save = idx_scene_save + 1
                HR_LF = np.zeros((U, V, H, W, 3), dtype='single')
                HR_LF[:] = LF

                # save
                file_name = [str(sub_save_dir) + '/' + '%s' % file.split('.')[0] + '.h5']
                with h5py.File(file_name[0], 'w') as hf:
                    # 注：matlab在保存h5py文件，好像会倒序保存
                    # 为了和matlab生成的数据格式保持一致，这里也做了transpose
                    hf.create_dataset('LF', data=HR_LF.transpose((4, 0, 1, 2, 3)), dtype='single')
                    hf.close()
                    pass

                print('%d test samples have been generated\n' % (idx_scene_save))



if __name__ == '__main__':
    args = parse_args()

    main(args)


