import importlib
import torch
from option import args
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time


if __name__ == "__main__":
    args.model_name = 'Kalantari2016'
    args.angRes_in = 2
    args.angRes_out = 7
    args.patch_size = 64

    args.device = 'cuda:1'
    device = torch.device(args.device)
    if 'cuda' in args.device:
        torch.cuda.set_device(device)

    MODEL_PATH = args.model_name
    MODEL = importlib.import_module(MODEL_PATH)
    net = MODEL.get_model(args).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of Parameters: %.4fM' % (total / 1e6))

    input = torch.randn([1, 3, args.angRes_in, args.angRes_in, args.patch_size, args.patch_size]).to(device)
    GPU_cost = torch.cuda.memory_allocated(0)
    out = net(input, [])
    GPU_cost1 = (torch.cuda.memory_allocated(0) - GPU_cost) / 1024 / 1024 / 1024  # GB
    print('   GPU consumption: %.4fGB' % (GPU_cost1))

    input = torch.randn([1, 3, args.angRes_in, args.angRes_in, args.patch_size, args.patch_size]).to(device)
    flops = FlopCountAnalysis(net, (input, [])).total()
    # print('   Model Parameters:', parameter_count_table(net))
    print('   Number of FLOPs: %.5fG' % (flops / 1e9))

    # start = time.time()
    # for _ in range(100):
    #     input = torch.randn([1, 3, args.angRes_in, args.angRes_in, args.patch_size, args.patch_size]).to(device)
    #     output = net(input)
    # end = time.time()
    # print('   Time used: %.4fs' % ((end - start) / 100))