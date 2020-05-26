import warnings
import os
import argparse
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import VGGEncoder, Decoder
from torchvision.utils import save_image
import traceback
import torch
import torch.nn.functional as F
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image

warnings.simplefilter("ignore", UserWarning)

batch_size = 4  # 每批训练的图片数量
epoch = 5       # 训练的轮次
patch_size = 5  # style_swap时的补丁大小
gpu = 0         # GPU设备号
learning_rate = 1e-4    # 学习率
tv_weight = 1e-6        # tv损失的权重
save_dir = "result"     # 训练生成的模型的文件夹
train_content_dir = "pic\\train_content"  # 默认训练的内容图像文件夹
train_style_dir = "pic\\train_style"      # 默认训练的风格图像文件夹
test_content_image = ''                   # 测试的内容图像
test_style_image = ''                     # 测试的风格图像
default_output_name = ''                  # 输出的结果名称
model_state_path = 'model_state.pth'      # 加载的模型参数

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

class PreprocessDataset(Dataset):
    def __init__(self, content_dir, style_dir, transforms=trans):
        content_dir_resized = content_dir + '_resized'
        style_dir_resized = style_dir + '_resized'
        if not (os.path.exists(content_dir_resized) and
                os.path.exists(style_dir_resized)):
            os.mkdir(content_dir_resized)
            os.mkdir(style_dir_resized)
            self._resize(content_dir, content_dir_resized)
            self._resize(style_dir, style_dir_resized)
        content_images = glob.glob((content_dir_resized + '/*'))
        np.random.shuffle(content_images)
        style_images = glob.glob(style_dir_resized + '/*')
        np.random.shuffle(style_images)
        self.images_pairs = list(zip(content_images, style_images))
        self.transforms = transforms

    @staticmethod
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(os.listdir(source_dir)):
            filename = os.path.basename(i)
            try:
                image = io.imread(os.path.join(source_dir, i))
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
                    if H > W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(target_dir, filename), image)
            except:
                continue

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_image, style_image = self.images_pairs[index]
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
        return content_image, style_image



# TV损失，使得像素之间的过渡更柔和
def TVloss(img, tv_weight):
    w_variance = torch.sum(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2))
    h_variance = torch.sum(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


def style_swap(content_feature, style_feature, kernel_size, stride=1):
    # content_feature 和 style_feature 的形状 (1, C, H, W)  ->  (1,256,64,64)
    # 这里的kernel_size等价于提取的补丁大小(从style_feature中提取补丁作为卷积核)
    # stride为步长
    kh, kw = kernel_size, kernel_size
    sh, sw = stride, stride

    patches = style_feature.unfold(2, kh, sh).unfold(3, kw,
                                                     sw)  # unfold(dim,size,step)   先把第三维展开，再把第四维展开，这样就能得到 所有的5*5补丁
    # (1,256,64,64)->(1,256,60,64,5)->(1,256,60,60,5,5)
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (1,60,60,256,5,5)
    patches = patches.reshape(-1, *patches.shape[-3:])  # (patch_numbers, C, kh, kw)  ->  (3600*256*5*5)

    # 计算弗洛贝尼乌斯范数并对每个过滤器的patch进行归一化处理
    norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)
    noramalized_patches = patches / norm

    conv_out = F.conv2d(content_feature, noramalized_patches)  # (1,3600,60,60)

    # 计算3600个卷积的结果在每个位置的最大值
    # 应该存在一个提供最大输出值的过滤器
    one_hots = torch.zeros_like(conv_out)
    one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

    # 反卷积
    deconv_out = F.conv_transpose2d(one_hots, patches)

    # 计算重叠的部分（用全为1的卷积核与原one_hots作反卷积，每个位置的值就是重叠的计数）
    overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

    # 除以计数就能取平均
    res = deconv_out / overlap
    return res


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def train():
    # 储存结果的文件夹，如果不存在则创建
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model_state_dir = f'{save_dir}/model_state'

    if not os.path.exists(model_state_dir):
        os.mkdir(model_state_dir)

    # 选择使用的设备，GPU或者CPU
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f'cuda:{gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    print(f'# Minibatch-size: {batch_size}')
    print(f'# epoch: {epoch}')
    print('')

    # 加载数据
    train_dataset = PreprocessDataset(train_content_dir, train_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型和优化器的载入
    encoder = VGGEncoder().to(device)
    decoder = Decoder().to(device)
    optimizer = Adam(decoder.parameters(), lr=learning_rate)

    # 开始训练
    criterion = nn.MSELoss()  # 均方误差
    loss_list = []

    # 第e轮训练
    for e in range(1, epoch + 1):
        print(f'Start {e} epoch')
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            content = content.to(device)  # (batch_size,3,256,256)
            style = style.to(device)  # (batch_size,3,256,256)
            content_feature = encoder(content)  # (batch_size,256,64,64)
            style_feature = encoder(style)  # (batch_size,256,64,64)

            style_swap_res = []
            for b in range(content_feature.shape[0]):
                c = content_feature[b].unsqueeze(0)  # (1,256,64,64)
                s = style_feature[b].unsqueeze(0)
                cs = style_swap(c, s, patch_size, 1)
                style_swap_res.append(cs)
            style_swap_res = torch.cat(style_swap_res, 0)  # (4,256,64,64)

            out_style_swap = decoder(style_swap_res)  # 风格转换后的结果（转换后的特征解码）
            out_content = decoder(content_feature)  # 内容特征直接解码
            out_style = decoder(style_feature)  # 风格特征直接解码

            out_style_swap_latent = encoder(out_style_swap)  # 风格转换的结果再提取特征
            out_content_latent = encoder(out_content)  # 内容编码解码再编码
            out_style_latent = encoder(out_style)  # 风格编码解码再编码

            image_reconstruction_loss = criterion(content, out_content) + criterion(style, out_style)  # 图像损失：图像编码再解码的损失

            feature_reconstruction_loss = criterion(style_feature, out_style_latent) + \
                                          criterion(content_feature, out_content_latent) + \
                                          criterion(style_swap_res, out_style_swap_latent)  # 特征损失：特征解码再编码的损失

            tv_loss = TVloss(out_style_swap, tv_weight) + TVloss(out_content, tv_weight) \
                      + TVloss(out_style, tv_weight)  # tv损失：相邻像素的差异

            loss = image_reconstruction_loss + feature_reconstruction_loss + tv_loss

            loss_list.append(loss.item())

            # 优化器反向传播调整参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'[{e}/total {epoch} epoch],[{i} /'
                  f'total {round(iters / batch_size)} iteration]: {loss.item()}')
        torch.save(decoder.state_dict(), f'{model_state_dir}/{e}_epoch.pth')  # 每个epoch模型保存


def test():

    if(test_content_image=='' or test_style_image=='' ):
        print("Please input the content and style image!")
        return

    # 设置设备为GPU/cpu
    if torch.cuda.is_available() and gpu >= 0:
        device = torch.device(f'cuda:{gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
    # device = 'cpu'
    # 载入编码器和解码器
    encoder = VGGEncoder().to(device)
    decoder = Decoder()
    decoder.load_state_dict(torch.load(model_state_path))
    decoder = decoder.to(device)

    try:
        content = Image.open(test_content_image)
        style = Image.open(test_style_image)
        c_tensor = trans(content).unsqueeze(0).to(device)
        s_tensor = trans(style).unsqueeze(0).to(device)
        # 不记录torch自带的计算图
        with torch.no_grad():
            cf = encoder(c_tensor)
            sf = encoder(s_tensor)
            style_swap_res = style_swap(cf, sf, patch_size, 1)
            del cf
            del sf
            del encoder
            out = decoder(style_swap_res)

        c_denorm = denorm(c_tensor, device)
        out_denorm = denorm(out, device)
        res = torch.cat([c_denorm, out_denorm], dim=0)
        res = res.to('cpu')
    except RuntimeError as e:
        traceback.print_exc()
        print('Images are too large to transfer.')

    if default_output_name == '':
        c_name = os.path.splitext(os.path.basename(test_content_image))[0]
        s_name = os.path.splitext(os.path.basename(test_style_image))[0]
        output_name = f'{c_name}_{s_name}'
    else:
        output_name = default_output_name

    try:
        save_image(out_denorm, f'{output_name}.jpg')
        save_image(res, f'{output_name}_pair.jpg', nrow=2)
        o = Image.open(f'{output_name}_pair.jpg')
        style = style.resize((i // 4 for i in content.size))
        box = (o.width // 2, o.height - style.height)
        o.paste(style, box)
        o.save(f'{output_name}_style_transfer_demo.jpg', quality=95)
        print(f'result saved into files starting with {output_name}')
    except:
        pass


if __name__ == '__main__':
    # 读取从控制台输入的参数
    parser = argparse.ArgumentParser(description='Style Swap by Pytorch')
    parser.add_argument('--action', '-a', required=True, type=str)
    parser.add_argument('--train_content_dir', default=train_content_dir, type=str)
    parser.add_argument('--train_style_dir', default=train_style_dir, type=str)
    parser.add_argument('--save_dir', default=save_dir, type=str)
    parser.add_argument('--content', '-c', default=test_content_image, type=str)
    parser.add_argument('--style', '-s', default=test_style_image, type=str)
    parser.add_argument('--output_name', '-o', default=default_output_name, type=str)
    parser.add_argument('--model_state_path', default=model_state_path, type=str)

    args = parser.parse_args()

    train_content_dir = args.train_content_dir
    train_style_dir = args.train_style_dir
    save_dir = args.save_dir
    test_content_image = args.content
    test_style_image = args.style
    default_output_name = args.output_name
    model_state_path = args.model_state_path

    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()
    else:
        print("no action!")
