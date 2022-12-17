from torch import nn

# CNN Discriminator for GAN
class CNN_Discriminator(nn.Module):
    def __init__(self, opt):
        super(CNN_Discriminator, self).__init__()

        self.ndf = opt.ndf
        # DCGAN定义的判别器，生成器没有池化层
        self.Discrim = nn.Sequential(
            # 卷积层
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，此处设定filer过滤器有64个，输出通道自然也就是64。
            # 因为对图片作了灰度处理，此处通道数为1，
            # 卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 3, 96, 96),bitch_size = 单次训练的样本量
            # output:(bitch_size, ndf, 32, 32), (96 - 5 +2 *1)/3 + 1 =32
            # LeakyReLu= x if x>0 else nx (n为第一个函数参数)，开启inplace（覆盖）可以节省内存，取消反复申请内存的过程
            # LeakyReLu取消了Relu的负数硬饱和问题，是否对模型优化有效有待考证
            nn.Conv2d(in_channels=3, out_channels= self.ndf, kernel_size= 5, stride= 3, padding= 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            # input:(ndf, 32, 32)
            nn.Conv2d(in_channels= self.ndf, out_channels= self.ndf * 2, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *2, 16, 16)
            nn.Conv2d(in_channels= self.ndf * 2, out_channels= self.ndf *4, kernel_size= 4, stride= 2, padding= 1,bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *4, 8, 8)
            nn.Conv2d(in_channels= self.ndf *4, out_channels= self.ndf *8, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf *8),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *8, 4, 4)
            # output:(1, 1, 1)
            nn.Conv2d(in_channels= self.ndf *8, out_channels= 1, kernel_size= 4, stride= 1, padding= 0, bias=True),

            # 调用sigmoid函数解决分类问题
            # 因为判别模型要做的是二分类，故用sigmoid即可，因为sigmoid返回值区间为[0,1]，
            # 可作判别模型的打分标准
            nn.Sigmoid()
        )

    def forward(self, x):
        # 展平后返回
        return self.Discrim(x).view(-1)