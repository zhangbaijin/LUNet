# import numpy as np

# import torch
# import torchvision
# from PIL import Image
# from torchvision import transforms as T

# import matplotlib.pyplot as plt
# from DGUNet_plus import DGUNet
# import torch
# import torchvision

# #feature_extractor = torchvision.models.resnet34(pretrained=True)
# model_restoration=DGUNet()
# model_restoration.cuda()


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# class SaveOutput:
# 	def __init__(self):
# 		self.outputs = []
# 	def __call__(self, module, module_in, module_out):
# 		self.outputs.append(module_out)
# 	def clear(self):
# 		self.outputs=[]
		
# save_output = SaveOutput()

# hook_handles = []

# for layer in model_restoration.modules():
# 	if isinstance(layer, torch.nn.Conv2d):
# 		handle = layer.register_forward_hook(save_output)
# 		hook_handles.append(handle)

# from PIL import Image
# from torchvision import transforms as T

# image = Image.open('./Datasets/MSRB_snow/test/input/test (1).png')
# transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
# X = transform(image).unsqueeze(dim=0).to(device)

# out = model_restoration(X)

# print(len(save_output.outputs))
# a_list = [0, 1, 6, 15, 28, 35]
# for i in a_list:
#     print(save_output.outputs[i].cpu().detach().squeeze(0).shape)

    
# def grid_gray_image(imgs, each_row: int):
#     '''
#     imgs shape: batch * size (e.g., 64x32x32, 64 is the number of the gray images, and (32, 32) is the size of each gray image)
#     '''
#     row_num = imgs.shape[0]//each_row
#     for i in range(row_num):
#         img = imgs[i*each_row]
#         img = (img - img.min()) / (img.max() - img.min())
#         for j in range(1, each_row):
#             tmp_img = imgs[i*each_row+j]
#             tmp_img = (tmp_img - tmp_img.min()) / (tmp_img.max() - tmp_img.min())
#             img = np.hstack((img, tmp_img))
#         if i == 0:
#             ans = img
#         else:
#             ans = np.vstack((ans, img))
#     return ans

# img0 = save_output.outputs[0].cpu().detach().squeeze(0)
# img0 = grid_gray_image(img0.numpy(), 8)
# img1 = save_output.outputs[1].cpu().detach().squeeze(0)
# img1 = grid_gray_image(img1.numpy(), 8)
# img6 = save_output.outputs[6].cpu().detach().squeeze(0)
# img6 = grid_gray_image(img6.numpy(), 8)
# img15 = save_output.outputs[15].cpu().detach().squeeze(0)
# img15 = grid_gray_image(img15.numpy(), 16)
# img29 = save_output.outputs[28].cpu().detach().squeeze(0)
# img29 = grid_gray_image(img29.numpy(), 16)

# plt.figure(figsize=(15, 15))
# plt.imshow(img0, cmap='gray')
import torch

from DGUNet_plus import DGUNet

PATH = './checkpoint/De-snow-min/models/DGUNet/model_epoch_250.pth'
# torch.save(cnn.state_dict(), PATH)
# cnn = CNNClass(*args, **kwargs)
cnn =DGUNet()
cnn.load_state_dict(torch.load(PATH))
cnn.eval()


class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


img = next(iter(train_data))[0]
fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0, wspace=0.2)
for i in range(len(cnn.features)):
    conv_out = LayerActivations(cnn.features, i)
    # o = cnn(Variable(img.cuda()))
    o = cnn(Variable(img))
    conv_out.remove()  #
    act = conv_out.features
    for j in range(1):
        ax = fig.add_subplot(5, 3, i + 1, xticks=[], yticks=[])
        ax.imshow(act[0][j].detach().numpy(), cmap="gray")

plt.savefig(r'./out/cnn_layer_visiual.png',dpi=600)
plt.show()
