from torch import nn

# CNN Generator for GAN
class CNN_Generator(nn.Module):
    # 构建初始化函数，传入opt类
    def __init__(self, opt):
        super(CNN_Generator, self).__init__()
        # self.ngf生成器特征图数目
        self.ngf = opt.ngf
        self.Gene = nn.Sequential(
            # 假定输入为opt.nz*1*1维的数据，opt.nz维的向量
            # output = (input - 1)*stride + output_padding - 2*padding + kernel_size
            # 把一个像素点扩充卷积，让机器自己学会去理解噪声的每个元素是什么意思。
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias =False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),

            # 输入一个4*4*ngf*8
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            # 输入一个8*8*ngf*4
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),

            # 输入一个16*16*ngf*2
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            # 输入一张32*32*ngf
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias =False),

            # Tanh收敛速度快于sigmoid,远慢于relu,输出范围为[-1,1]，输出均值为0
            nn.Tanh(),

        )# 输出一张96*96*3

    def forward(self, x):
        return self.Gene(x)