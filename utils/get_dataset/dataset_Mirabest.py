import torch

from torchvision import transforms


from data.MiraBest import MiraBest


# def get_MiraBest_binary(batch_size = 16, downscale = True):
#     transform = transforms.Compose(
#     [transforms.ToTensor()])
# 
#     #resize from 150x150 to 50x50
#     if downscale:
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((30,30)), 
#             
#         ])
# 
#     trainset = MiraBest(root='./dataMirabest', train=True, download=True, transform=transform)  
#     #print(len(trainset))
#     batch_size_train = batch_size
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True)
# 
#     testset = MiraBest(root='./dataMirabest', train=False, download=True, transform=transform) 
#     #print(len(testset))
#     batch_size_test = batch_size
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)
# 
#     info = "MiraBest dataset for binary classification.\n"
# 
#     return trainloader, testloader, info