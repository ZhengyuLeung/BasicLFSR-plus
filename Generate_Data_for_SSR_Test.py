import argparse
import os
import h5py
import numpy as np
from pathlib import Path
import scipy.io as scio
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='SSR', help='SSR, ASR')
    parser.add_argument("--max_angRes", type=int, default=9, help="angular resolution")
    parser.add_argument('--data_for', type=str, default='test', help='')
    parser.add_argument('--src_data_path', type=str, default='../datasets/', help='')
    parser.add_argument('--save_data_path', type=str, default='./', help='')
    return parser.parse_args()


def main(args):
    ''' 为了保证低分的感受野是32*32，需要根据 angRes 和 scale_factor 计算 patchsize 和 stride '''
    angRes = args.max_angRes

    ''' 建立保存路径 '''
    save_dir = Path(args.save_data_path + 'data_for_' + args.data_for)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath(str(args.task) + '_' + str(angRes) + 'x' + str(angRes))
    save_dir.mkdir(exist_ok=True)


    ''' 遍历光场数据集 '''
    src_datasets = os.listdir(args.src_data_path)
    src_datasets.sort()
    for index_dataset in range(len(src_datasets)):
        if src_datasets[index_dataset] not in ['EPFL', 'HCI_new', 'HCI_old', 'INRIA_Lytro', 'Stanford_Gantry', 'Stanford_Lytro']:
            continue
        idx_save = 0
        name_dataset = src_datasets[index_dataset]
        ''' 建立数据集对应的保存路径 '''
        sub_save_dir = save_dir.joinpath(name_dataset)
        sub_save_dir.mkdir(exist_ok=True)



        ''' 遍历数据集下的场景 '''
        src_sub_dataset = args.src_data_path + name_dataset + '/' + args.data_for + '/'
        for root, dirs, files in os.walk(src_sub_dataset):
            for file in files:
                idx_scene_save = 0
                print('Generating training data of Scene_%s in Dataset %s......\t' %(file, name_dataset))
                try:
                    data = h5py.File(root + file, 'r')
                    # 加载光场数据
                    LF = np.array(data[('LF')]).transpose((4, 3, 2, 1, 0))
                except:
                    data = scio.loadmat(root + file)
                    LF = np.array(data['LF'])


                (U, V, H, W, _) = LF.shape
                H = H // 4 * 4
                W = W // 4 * 4

                # Extract central angRes * angRes views
                LF = LF[(U-angRes)//2:(U+angRes)//2, (V-angRes)//2:(V+angRes)//2, 0:H, 0:W, 0:3]
                LF = LF.astype('double')
                (U, V, H, W, _) = LF.shape

                # 测试集不对 光场数据 切割 patches
                idx_save = idx_save + 1
                idx_scene_save = idx_scene_save + 1
                HR_LF = np.zeros((U, V, H, W, 3), dtype='single')
                HR_LF[:] = LF

                # 保存路径
                file_name = [str(sub_save_dir) + '/' + '%s' % file.split('.')[0] + '.h5']
                with h5py.File(file_name[0], 'w') as hf:
                    # 注：matlab在保存h5py文件，好像会倒序保存
                    # 为了和matlab生成的数据格式保持一致，这里也做了transpose
                    hf.create_dataset('LF', data=HR_LF.transpose((4, 0, 1, 2, 3)), dtype='single')
                    hf.close()
                    pass

                #
                print('%d training samples have been generated\n' % (idx_scene_save))

                pass
            pass
        pass

    pass



if __name__ == '__main__':
    args = parse_args()

    main(args)