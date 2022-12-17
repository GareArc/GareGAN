import os

class Config(object):
    """
    Configuration for GAN training.
    """
    # 0.参数调整
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets")
    virs = "result"
    num_workers = 4  # 多线程
    img_size = 96  # 剪切图片的像素大小
    batch_size = 256  # 批处理数量
    max_epoch = 20   # 最大轮次
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5  # 正则化系数，Adam优化器参数
    gpu = False  # 是否使用GPU运算（建议使用）
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的卷积核个数
    ndf = 64  # 判别器的卷积核个数

    # 1.模型保存路径
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs")  # opt.netg_path生成图片的保存路径
    # 判别模型的更新频率要高于生成模型
    d_every = 1  # 每一个batch 训练一次判别器
    g_every = 5  # 每1个batch训练一次生成模型
    save_every = 5  # 每save_every次保存一次模型
    prev_model = False # 是否有model直接加载
    netd_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"netdfile")
    netg_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"netgfile")
    netd_load_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"netdfile", "netd_19.pth")
    netg_load_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),"netgfile", "netg_19.pth")

    # 测试数据
    gen_img = "result.png"
    # 选择保存的照片
    # 一次生成保存64张图片
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0    # 生成模型的噪声均值
    gen_std = 1     # 噪声方差