# import torch

# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# torch.zeros(1).cuda()
# print(torch.cuda.get_arch_list())
import dgl
import torch as th
u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
g = dgl.graph((u, v))
g.ndata['x'] = th.randn(5, 3)   # 原始特征在CPU上
g.device
cuda_g = g.to('cuda:0')         # 接受来自后端框架的任何设备对象
cuda_g.device
cuda_g.ndata['x'].device        # 特征数据也拷贝到了GPU上
# 由GPU张量构造的图也在GPU上
u, v = u.to('cuda:0'), v.to('cuda:0')
g = dgl.graph((u, v))
g.device
cuda_g.in_degrees()
cuda_g.in_edges([2, 3, 4])                          # 可以接受非张量类型的参数
cuda_g.in_edges(th.tensor([2, 3, 4]).to('cuda:0'))  # 张量类型的参数必须在GPU上
cuda_g.ndata['h'] = th.randn(5, 4)                  # ERROR! 特征也必须在GPU上！

# wget https://pypi.tuna.tsinghua.edu.cn/packages/37/42/0132befd0d7c24e9c4cf126b4051a9145e4c54eee474b9ac5de02b95bb33/dgl_cu111-0.6.1-cp37-cp37m-manylinux1_x86_64.whl#sha256=6ac8adfb38bdb6214f7c527b38d838b25e01ca161971c8075dda323ade5fc48e
# pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113