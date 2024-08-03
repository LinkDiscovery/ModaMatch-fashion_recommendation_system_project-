from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

#이미지 불러오는코드
response = requests.get("http://172.21.96.67:8080/img/ACK1VT002.jpg")
img = Image.open(BytesIO(response.content))
#------------------

#img = Image.open("http://172.21.96.67:8080/img/ACK1VT002.jpg")

print(img.size)

plt.imshow(img)
plt.show()