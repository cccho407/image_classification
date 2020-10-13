from torchvision import datasets, models, transforms
# import mb2
import cv2
import numpy as np
import torch
import requests
import os
import shutil
from PIL import Image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('example/model/savetest.pth', map_location=DEVICE)

count_ballon, count_banana, count_bell, count_cdplayer, count_cleaver, count_cradle, count_crane, count_daisy,\
count_helmet, count_speaker= 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

dir = "example/test"
# expect = 0
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
    device = 'cuda'
else:
    device = 'cpu'

# correct_count = 0
file_list = os.listdir(dir)
file_list_jpeg = [file for file in file_list if file.endswith(".JPEG")]

for image in file_list_jpeg:

    orig_image1 = cv2.imread(dir+'/'+image)

    to_pil = transforms.ToPILImage()
    orig_image = to_pil(orig_image1)
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    text_image = image

    image = trans(orig_image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        result = model(image)
    print("-"*50)
    # print(result)
    pr = torch.argmax(torch.nn.functional.softmax(result[0], dim=0))
    result1 = torch.nn.functional.softmax(result[0], dim=0)
    # print(result1)
    round_result = round(float(result1[pr]), 4)  # Decimal point to round
    print(f"conf : {round_result}, result : {pr}")
    
    # if pr == expect :
    #    correct_count +=1
    src = dir + '/' + text_image
    if pr == 0:
        dst = "example/test/ballon/"+text_image
        shutil.copyfile(src, dst)
        count_ballon += 1
    elif pr == 1:
        dst = "example/test/banana/" + text_image
        shutil.copyfile(src, dst)
        count_banana += 1
    elif pr == 2:
        dst = "example/test/bell/" + text_image
        shutil.copyfile(src, dst)
        count_bell += 1
    elif pr == 3:
        dst = "example/test/cdplayer/" + text_image
        shutil.copyfile(src, dst)
        count_cdplayer += 1
    elif pr == 4:
        dst = "example/test/cleaver/" + text_image
        shutil.copyfile(src, dst)
        count_cleaver += 1
    elif pr == 5:
        dst = "example/test/cradle/" + text_image
        shutil.copyfile(src, dst)
        count_cradle += 1
    elif pr == 6:
        dst = "example/test/crane/" + text_image
        shutil.copyfile(src, dst)
        count_crane += 1
    elif pr == 7:
        dst = "example/test/daisy/" + text_image
        shutil.copyfile(src, dst)
        count_daisy += 1
    elif pr == 8:
        dst = "example/test/helmet/" + text_image
        shutil.copyfile(src, dst)
        count_helmet += 1
    elif pr == 9:
        dst = "example/test/speaker/" + text_image
        shutil.copyfile(src, dst)
        count_speaker += 1


print(f"ballon acc : {count_ballon}/100")
print(f"banana acc : {count_banana}/100")
print(f"bell acc : {count_bell}/100")
print(f"cdplayer acc : {count_cdplayer}/100")
print(f"cleaver acc : {count_cleaver}/100")
print(f"cradle acc : {count_cradle}/100")
print(f"crane acc : {count_crane}/100")
print(f"daisy acc : {count_daisy}/100")
print(f"helmet acc : {count_helmet}/100")
print(f"speaker acc : {count_speaker}/100")
