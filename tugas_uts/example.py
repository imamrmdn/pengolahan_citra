import cv2 
import matplotlib.pyplot as pyp


#membaca gambar di folder img
img = cv2.imread("img/image.jpg") 

#membuat gmbr menjadi BRG dan RGB dan skala abu abu
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

#menggunakan haas cascade classifier untuk deteksi objek
data = cv2.CascadeClassifier('data.xml')
found = data.detectMultiScale(img_gray, 
								minSize =(20, 20)) 

amount_found = len(found) 
if amount_found != 0: 
	for (x, y, width, height) in found: 

		#menandai object atau gambar dengan persegi merah
		cv2.rectangle(img_rgb, (x, y), 
					(x + height, y + width), 
					(255, 0, 0), 5) 

#menampilkan object yang sudah ditandai	
pyp.title('Tugas UTS Mendeteksi Object')	
pyp.subplot(1, 1, 1) 
pyp.imshow(img_rgb) 
pyp.show() 

cv2.waitKey(0)
cv2.destroyAllWindows()