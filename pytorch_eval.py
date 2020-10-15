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

counter= 0
classes = ['ballon', 'banana', 'bell', 'cdplayer', 'cleaver', 'cradle', 'crane', 'daisy', 'helmet', 'speaker']
class_correct = list(0. for i in range(10))

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
    # for get a total acc
    if round_result<0.6:
        counter += 1
    class_correct[pr] += 1
    src = dir + '/' + text_image
    dst = "example/test/"+classes[pr]+'/'+text_image
    shutil.copyfile(src, dst)
    
total_acc = (1000-counter)/1000*100
for i in range(10):
    print(f'Accuracy of {classes[i]} : {class_correct[i]}/100')
print(f"Estimated number of misplaced images: {counter}")
print(f"total acc : {total_acc}%")

