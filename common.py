import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from utils.utils import *
from utils.utils_datasets import TrainSetDataLoader, MultiTestSetDataLoader
from collections import OrderedDict
import imageio
from utils.MultiDegradation import LF_Blur, random_crop_SAI, LF_Bicubic, LF_Noise


def main(args):
    ''' Create Dir for Save'''
    log_dir, checkpoints_dir, val_dir = create_dir(args)

    ''' Logger '''
    logger = Logger(log_dir, args)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA Training LOADING '''
    logger.log_string('\nLoad Training Dataset ...')
    train_Dataset = TrainSetDataLoader(args)
    logger.log_string("The number of training data is: %d" % len(train_Dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_Dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True,)

    ''' DATA Validation LOADING '''
    logger.log_string('\nLoad Validation Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    logger.log_string("The number of validation data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    logger.log_string('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)


    ''' Load Pre-Trained PTH '''
    if args.use_pre_ckpt == False:
        net.apply(MODEL.weights_init)
        start_epoch = 0
        logger.log_string('Do not use pre-trained model!')
    else:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            start_epoch = checkpoint['epoch']
            try:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k  # add `module.`
                    new_state_dict[name] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
            except:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    new_state_dict[k] = v
                # load params
                net.load_state_dict(new_state_dict)
                logger.log_string('Use pretrain model!')
        except:
            net = MODEL.get_model(args)
            net.apply(MODEL.weights_init)
            start_epoch = 0
            logger.log_string('No existing model, starting training from scratch...')
            pass
        pass
    start_epoch = args.start_epoch
    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    logger.log_string('PARAMETER ...')
    logger.log_string(args)


    ''' LOSS LOADING '''
    criterion = MODEL.get_loss(args).to(device)
    if args.use_pre_ckpt:
        try:
            ckpt_path = args.path_pre_pth
            checkpoint = torch.load(ckpt_path, map_location='cpu')  # 存放模型的位置
            criterion_state_dict = OrderedDict()
            for k, v in checkpoint['loss_dict'].items():
                criterion_state_dict[k] = v
            # load params
            criterion.load_state_dict(criterion_state_dict)
            logger.log_string('Use pretrain criterion!')
        except:
            pass

    ''' Optimizer '''
    optimizer = torch.optim.Adam(
        [paras for paras in net.parameters() if paras.requires_grad == True],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)


    ''' TRAINING & TEST '''
    logger.log_string('\nStart training...')
    for idx_epoch in range(start_epoch, args.epoch):
        logger.log_string('\nEpoch %d /%s:' % (idx_epoch + 1, args.epoch))

        ''' Training '''
        loss_epoch_train = train(train_loader, device, net, criterion, optimizer)
        logger.log_string('The %dth Train, loss is: %.5f' % (idx_epoch + 1, loss_epoch_train))

        ''' Save PTH  '''
        if args.local_rank == 0:
            if args.task == 'SSR':
                save_ckpt_path = str(checkpoints_dir) + '/%s_%dx_%s_epoch_%02d_model.pth' % (args.model_name,
                    args.scale_factor, args.task, idx_epoch + 1)
            elif args.task == 'ASR':
                save_ckpt_path = str(checkpoints_dir) + '/%s_%dx%d_%dx%d_%s_epoch_%02d_model.pth' % (args.model_name,
                    args.angRes_in, args.angRes_in, args.angRes_out, args.angRes_out, args.task, idx_epoch + 1)

            try:
                state = {
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                    'loss_dict': criterion.module.sstate_dict() if hasattr(criterion,
                                                                           'module') else criterion.state_dict(),
                }
            except:
                state = {
                    'epoch': idx_epoch + 1,
                    'state_dict': net.module.state_dict() if hasattr(net, 'module') else net.state_dict(),
                }
            torch.save(state, save_ckpt_path)
            logger.log_string('Saving the epoch_%02d model at %s' % (idx_epoch + 1, save_ckpt_path))


        ''' Validation '''
        step = 5
        if idx_epoch%step == 0 or idx_epoch > 70:
            with torch.no_grad():
                ''' Create Excel for PSNR/SSIM '''
                excel_file = ExcelFile()

                psnr_list = []
                ssim_list = []
                lpips_list = []
                for index, test_name in enumerate(test_Names):
                    test_loader = test_Loaders[index]

                    epoch_dir = val_dir.joinpath('VAL_epoch_%02d' % (idx_epoch + 1))
                    epoch_dir.mkdir(exist_ok=True)
                    save_dir = epoch_dir.joinpath(test_name)
                    save_dir.mkdir(exist_ok=True)

                    psnr, ssim, lpips = test(args, test_name, test_loader, net, excel_file, save_dir)
                    excel_file.write_sheet(test_name, 'Average', 'PSNR', psnr)
                    excel_file.write_sheet(test_name, 'Average', 'SSIM', ssim)
                    excel_file.write_sheet(test_name, 'Average', 'LPIPS', lpips)
                    excel_file.add_count(2)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
                    lpips_list.append(lpips)

                    logger.log_string('The %dth Test on %s, psnr/ssim/lpips is %.2f/%.4f/%.4f' % (
                    idx_epoch + 1, test_name, psnr, ssim, lpips))
                    pass
                psnr_mean = np.array(psnr_list).mean()
                ssim_mean = np.array(ssim_list).mean()
                lpips_mean = np.array(lpips_list).mean()
                excel_file.write_sheet('ALL', 'Average', 'PSNR', psnr_mean)
                excel_file.write_sheet('ALL', 'Average', 'SSIM', ssim_mean)
                excel_file.write_sheet('ALL', 'Average', 'LPIPS', lpips_mean)
                logger.log_string('The mean psnr on testsets is %.5f, mean ssim is %.4f, mean lpips is %.4f' 
                    % (psnr_mean, ssim_mean, lpips_mean))
                excel_file.xlsx_file.save(str(epoch_dir) + '/evaluation_%s_%s.xlsx' % (args.task, args.model_name))
                pass
            pass

        ''' scheduler '''
        scheduler.step()
        try:
            criterion.GAN_loss.scheduler.step()
        except:
            pass
    pass


def train(train_loader, device, net, criterion, optimizer):
    ''' training one epoch '''
    loss_list = []

    # set the degradation function
    blur_func = LF_Blur(
        kernel_size=args.blur_kernel,
        blur_type=args.blur_type,
        sig_min=args.sig_min, sig_max=args.sig_max,
        lambda_min=args.lambda_min, lambda_max=args.lambda_max,
    )

    for idx_iter, (LF, img) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=70):
        ''' degradation '''
        if args.task == 'SSR':
            # Isotropic or Anisotropic Gaussian Blurs
            [LF_degraded, kernels, sigmas] = blur_func(LF)

            # down-sampling
            LF_degraded = LF_Bicubic(LF_degraded, scale=1/args.scale_factor)

            LF_input = LF_degraded
            LF_target = LF
            info = None
        elif args.task == 'ASR':
            angFactor = (args.angRes_out - 1) // (args.angRes_in - 1)
            LF_sampled = LF[:, :, ::angFactor, ::angFactor, :, :]
            LF, LF_sampled = random_crop_SAI(LF, LF_sampled, SAI_patch=args.patch_for_train * args.scale_factor)

            LF_input = LF_sampled
            LF_target = LF
            info = None

        ''' super-resolve the degraded LF images'''
        LF_input = LF_input.to(device)      # low resolution
        LF_target = LF_target.to(device)    # high resolution
        net.train()
        LF_out = net(LF_input, info)
        loss = criterion(LF_out, LF_target, info)

        ''' calculate loss '''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.data.cpu())
        pass

    loss_mean = float(np.array(loss_list).mean())
    return loss_mean


def test(args, test_name, test_loader, net, excel_file, save_dir=None):
    psnr_list = []
    ssim_list = []

    # set the degradation function
    blur_func = LF_Blur(
        kernel_size=args.blur_kernel,
        blur_type=args.blur_type,
        sig=args.sig,
        lambda_1=args.lambda_1, lambda_2=args.lambda_2,
    )

    for idx_iter, (LF, LF_name) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=70):
        ''' degradation '''
        if args.task == 'SSR':
            # Isotropic or Anisotropic Gaussian Blurs
            [LF_degraded, kernels, sigmas] = blur_func(LF)

            # down-sampling
            LF_degraded = LF_Bicubic(LF_degraded, scale=1/args.scale_factor)

            LF_input = LF_degraded
            LF_target = LF
            info = None
        elif args.task == 'ASR':
            angFactor = (args.angRes_out_for_test - 1) // (args.angRes_in_for_test - 1)
            LF_sampled = LF[:, :, ::angFactor, ::angFactor, :, :]

            LF_input = LF_sampled
            LF_target = LF
            info = None

        ''' Crop LFs into Patches '''
        LF_divide_integrate_func = LF_divide_integrate(args.scale_factor, args.patch_size_for_test, args.stride_for_test)
        sub_LF_input = LF_divide_integrate_func.LFdivide(LF_input)

        ''' SR the Patches '''
        sub_LF_out = []
        for i in range(0, sub_LF_input.size(0), args.minibatch_for_test):
            tmp = sub_LF_input[i:min(i + args.minibatch_for_test, sub_LF_input.size(0)), :, :, :, :, :]
            with torch.no_grad():
                net.eval()
                torch.cuda.empty_cache()
                out = net(tmp.to(args.device), info)
                sub_LF_out.append(out['SR'])
        sub_LF_out = torch.cat(sub_LF_out, dim=0)
        LF_out = LF_divide_integrate_func.LFintegrate(sub_LF_out).unsqueeze(0)
        LF_out = LF_out[:, :, :, :, 0:LF_target.size(-2), 0:LF_target.size(-1)].cpu().detach()

        ''' Calculate the PSNR & SSIM '''
        psnr, ssim, lpips = cal_metrics(args, LF_target, LF_out, bd=0 if 'SSR' in args.task else 22)
        excel_file.write_sheet(test_name, LF_name[0], 'PSNR', psnr)
        excel_file.write_sheet(test_name, LF_name[0], 'SSIM', ssim)
        excel_file.write_sheet(test_name, LF_name[0], 'LPIPS', lpips)
        excel_file.add_count(1)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        lpips_list.append(lpips)


        ''' Save RGB '''
        if save_dir is not None:
            save_dir_ = save_dir.joinpath(LF_name[0])
            save_dir_.mkdir(exist_ok=True)
            views_dir = save_dir_.joinpath('views')
            views_dir.mkdir(exist_ok=True)

            # save the error image of center view
            if args.task == 'ASR':
                path = str(save_dir_) + '/' + LF_name[0] + '_error.bmp'
                error = (LF_rgb2ycbcr(LF_target) - LF_rgb2ycbcr(LF_out))
                error = error[0, 0, args.angRes_out_for_test//2, args.angRes_out_for_test//2, :, :]
                plt.imsave(path, error.clamp(0, 0.1), cmap=plt.cm.jet)


            # save the center view
            LF_out = (LF_out.squeeze(0).permute(1, 2, 3, 4, 0).cpu().detach().numpy().clip(0, 1) * 255).astype('uint8')
            path = str(save_dir_) + '/' + LF_name[0] + '_SAI.bmp'
            img = LF_out[args.angRes_out_for_test//2, args.angRes_out_for_test//2, :, :, :]
            imageio.imwrite(path, img)


            # # save all views
            # for i in range(args.angRes_out_for_test):
            #     for j in range(args.angRes_out_for_test):
            #         path = str(views_dir) + '/' + LF_name[0] + '_' + str(i) + '_' + str(j) + '.bmp'
            #         img = LF_out[i, j, :, :, :]
            #         imageio.imwrite(path, img)
            #     pass

        pass

    return [np.array(psnr_list).mean(), np.array(ssim_list).mean(), np.array(lpips_list).mean()]


def main_test(args):
    ''' Create Dir for Save '''
    _, _, result_dir = create_dir(args)
    result_dir = result_dir.joinpath('TEST')
    result_dir.mkdir(exist_ok=True)

    ''' CPU or Cuda'''
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    ''' DATA TEST LOADING '''
    print('\nLoad Test Dataset ...')
    test_Names, test_Loaders, length_of_tests = MultiTestSetDataLoader(args)
    print("The number of test data is: %d" % length_of_tests)


    ''' MODEL LOADING '''
    print('\nModel Initial ...')
    MODEL_PATH = 'model.' + args.task + '.' + args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args)

    ''' Load Pre-Trained PTH '''
    checkpoint = torch.load(args.path_pre_pth, map_location='cpu')
    try:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = 'module.' + k  # add `module.`
            new_state_dict[name] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
    except:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k] = v
        # load params
        net.load_state_dict(new_state_dict)
        print('Use pretrain model!')
        pass

    net = net.to(device)
    cudnn.benchmark = True

    ''' Print Parameters '''
    print('PARAMETER ...')
    print(args)

    ''' TEST on every dataset '''
    print('\nStart test...')
    with torch.no_grad():
        ''' Create Excel for PSNR/SSIM '''
        excel_file = ExcelFile()

        psnr_list = []
        ssim_list = []
        for index, test_name in enumerate(test_Names):
            test_loader = test_Loaders[index]

            save_dir = result_dir.joinpath(test_name)
            save_dir.mkdir(exist_ok=True)

            psnr, ssim, lpips = test(args, test_name, test_loader, net, excel_file, save_dir)
            excel_file.write_sheet(test_name, 'Average', 'PSNR', psnr)
            excel_file.write_sheet(test_name, 'Average', 'SSIM', ssim)
            excel_file.write_sheet(test_name, 'Average', 'LPIPS', lpips)
            excel_file.add_count(2)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            lpips_list.append(lpips)
            print('Test on %s, psnr/ssim/lpips is %.2f/%.4f/%.4f' % (test_name, psnr, ssim, lpips))
            pass

        psnr_mean = np.array(psnr_list).mean()
        ssim_mean = np.array(ssim_list).mean()
        lpips_mean = np.array(lpips_list).mean()
        excel_file.write_sheet('ALL', 'Average', 'PSNR', psnr_mean)
        excel_file.write_sheet('ALL', 'Average', 'SSIM', ssim_mean)
        excel_file.write_sheet('ALL', 'Average', 'LPIPS', lpips_mean)
        print('The mean psnr on testsets is %.2f, mean ssim is %.4f, mean lpips is %.4f' 
            % (psnr_mean, ssim_mean, lpips_mean))
        excel_file.xlsx_file.save(str(result_dir) + '/evaluation_%s_%s_%.1f_noise_%.1f.xlsx' % (args.task, args.model_name, args.sig, args.noise))
    pass


