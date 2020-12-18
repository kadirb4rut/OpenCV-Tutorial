# %% Computer Vision with OpenCV

"""
 - OpenCV is one of the most important Libraries for Computer Vision.

"""

import cv2 # OpenCV import etme

# %% Resim İçe Aktarma

img = cv2.imread("messi.jpeg",0) # --> 0 (Sıfır) resmi Gray Scale olarak içe aktarır.

# Görselleştir.

cv2.imshow("Ilk Resim",img)

cv2.destroyWindow("Ilk Resim") # Seçilen Resmi kapatır.

cv2.destroyAllWindows() # Açılan Bütün resimleri kapatır.

# Klavye bağlantısı kurmak ve açılan video veya resimleri ESC tuşu ile kapatmak.

"""
- NOT : Tuşları ASCII Değerleri ile tanıtıp pencereleri kapatma işlevi tanımlayacağız.
        Kapatmak için resim üzerindeyken ESC tuşuna basılmalıdır.
 
"""

k = cv2.waitKey(0) & 0xFF

if k == 27: # 27 ESC Tuşunun ASCII değeridir.!!!
    cv2.destroyAllWindows()
elif k == ord('s'): # s tuşuna basınca neler yapacağını böyle tanımlarız.
    cv2.imwrite("messi_gray.png", img)
    cv2.destroyAllWindows()
    
# %% Video İçe Aktarma
    
import cv2
import time

video_name = "MOT17-04-DPM.mp4"

# Video içe aktarma Capture

cap = cv2.VideoCapture(video_name)

# Video genişliği ve yüksekliği

print("Genişlik : ",cap.get(3)) # --> cap.get(3) Video Genişliği bulmak için
print("Yükseklik : ",cap.get(4))# --> cap.get(4) Video Yüksekliği bulmak için

"""
 * Mutlaka Video olup olmadığı veya açılıp açılmadığı tespit edilmelidir.
 * Video değilse bana bir hata döndürmesini istiyorum. Çünkü video boş bile
   olsa yükler ve hata yokmuş gibi çalışır. Fakat aslında programımız düzgün
   çalışmaz.
 
 * Hata döndüren İlgili kod aşağıdadır.
 
"""
if cap.isOpened() == False:
    print("Hata! Video açılamıyor.")
    
    
# Video'yu Okumak

while cap.isOpened() == True: # --> Alttaki işlem tek bir Frame yani Resim döndürür. Video olması için While döngüsü ile True olduğu sürece döndürmeliyiz.
    ret, frame = cap.read() 
    if ret == True:
        time.sleep(0.01) # Video çok hızlı oynayacağı için time ile yavaşlatma uygulandı.
        cv2.imshow("Video", frame)
    else: break

    if cv2.waitKey(1) & 0xFF == ord("q"): # Çıkmak istersek "q" tuşu ile çıkabiliriz.
        break
""" 
    ret ve frame adında iki değer döndürür. 
    ret bu işlemin başarılı olup olamdığını yani True veya False döndürür. 
  
 * Dönen True veya False değerine göre işlem yapmak için IF - ELSE sorgularıyla
   koda devam edilir.
                            
"""

cap.release() # Stop Video Capture
cv2.destroyAllWindows() # Bütün işlemler bittikten sonra pencereleri kapatıyoruz.


# %% Kamera Açmak ve Video Kaydetmek

import cv2
import time

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("{} * {} pixel".format(width,height))                       

# Video Kaydetme

# Define the codec and create VideoWriter object

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('Video_kaydi.mp4',fourcc, 24.0, (width,height))
   
while cap.isOpened() == True:
    
    ret, frame = cap.read()
    time.sleep(0.01)
    cv2.imshow("Video", frame)
    frame2 = cv2.flip(frame, 1) # --> Rotate yani AYNA ÖZELLİĞİ 
    cv2.imshow("Flipped Video", frame2)
    
    #save
    writer.write(frame2)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release() # Video okuyucuyu sonlandır.
writer.release() # Video Kaydetmeyi sonlandır.
cv2.destroyAllWindows() # Tüm açık pencereleri sonlandır.
    
    
# %% Yeniden Boyutlandırma ve Kırpma

import cv2

img = cv2.imread("lenna.png",0) # 0 --> Siyah Beyaz yapmak için kullanılır!!
cv2.imshow("lenna", img)

print("Resim Boyutu : ", img.shape)

img2 = cv2.imread("lenna.png") 
cv2.imshow("lennaRGB", img2) 

cv2.destroyAllWindows()

# Resized 

resized_img = cv2.resize(img,(800,800))
print("Yeniden boyutlandırılmış resim : ",resized_img.shape)
    
cv2.imshow("Resize Picture",resized_img)    

# Kırpma 

imgCropped = img[:200,:300]
cv2.imshow("Cropped Picture",imgCropped)   
    
# %% Şekil ve Metin Ekleme

import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8)
cv2.imshow("Siyah Resim",img)

# Line (Çizgi) Çizdirme
# cv2.line(resim, (baslangıc_pixel),(bitis_pixel),(B,G,R), Kalınlık)

cv2.line(img,(0,0),(512,512),(0,0,255), 3)
cv2.imshow("Picture with Line",img)

# Dikdörtgen (Rectangle) Çizdirme
# cv2.rectangle(resim, (baslangıc_pixel),(bitis_pixel),(B,G,R), Kalınlık)

cv2.rectangle(img,(0,0),(256,256),(0,255,0),3)
cv2.imshow("Dikdortgen",img)

# Dikdörtgenin içini doldurmak için kalınlık yerine cv2.FILLED parametresi eklenir!!!!

cv2.rectangle(img,(0,0),(256,256),(0,255,0), cv2.FILLED)
cv2.imshow("Dikdortgen",img)

# Çember (Circle) Çizdirme.
# cv2.circle(resim, merkez, yarıçap, renk)

cv2.circle(img,(300,300),50,(255,0,0))
cv2.imshow("Circle",img)

# Çemberin içi cv2.FILLED parametresi doldurularak Daire Şekli elde edilir!!!

cv2.circle(img,(300,300),50,(255,0,0),cv2.FILLED)
cv2.imshow("Daire",img)

# Yazı (Text) Eklemek
# cv2.putText(resim, başlangıç noktası, font, yazı kalınlığı, renk)

cv2.putText(img,"Resim",(350,350),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))
cv2.imshow("Add to Text",img)

# %% Görüntülerin Birleştirilmesi

import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow("Lenna",img)

# HORIZONTAL Birleştirme

hor = np.hstack((img,img)) # np.hstack() --> Horizontal Birleştirme
cv2.imshow("Horizontal_Lena",hor) 

# VERTICAL Birleştirme

ver = np.vstack((img,img))
cv2.imshow("Vertical_Lena", ver)

# %% Perspektif Çarpıtma / Düzeltme

import cv2
import numpy as np

img = cv2.imread("kart.png")
cv2.imshow("Original",img)

# 1. Adım --> Resmin Boyutunu belirle.

width = 400 
height = 500 

pts1 = np.float32([[203,1],[1,472],[540,150],[338,617]]) # Orjinal resmin köşeleri
pts2 = np.float32([[0,0],[0,width],[height,0],[height,width]])

"""
Transform (Çevirme) işlemini gerçekleştirmek için Perspektif Transform matrisi
cv2.getPerspectiveTransform() metodu ile elde edilir.!!!!

"""
matrix = cv2.getPerspectiveTransform(pts1,pts2)
print(matrix) 

# Çevirme İşlemi
# Nihai Dönüştürülmüş Resim

imgOutput = cv2.warpPerspective(img,matrix,(height,width))  
cv2.imshow("Nihai Resim",imgOutput)

# %% Görüntüleri Birbirleri ile Karıştırmak

import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.JPG")
img2 = cv2.imread("img2.JPG")

"""
 * NOT : OpenCV resimleri BGR formatında gösterdiği için resimler farklı 
         renklerde görünür. Yapılması gereken İşlem ise ;
         
         "BGR" formatındaki yüklenen resmi "RGB" formatına çevirmektir!!!!
         
         Bu işlem için :
         
         cv2.cvtColor(img, cvt.COLOR_BGR2RGB) --> METHODU KULLANILIR!!
    
"""

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # BGR uzayından RGB uzayına dönüşüm.
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # BGR uzayından RGB uzayına dönüşüm.

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#BİRLEŞTİRME İŞLEMİ İÇİN :

# 1. ADIM : Shape (Boyut) Control --> İki resmin Boyutları aynı olmak zorunda!!

print(img1.shape)
print(img2.shape) 

# İki resmi aynı boyuta getirme. Resize işlemi

img1 = cv2.resize(img1, (600,600))
img2 = cv2.resize(img2, (600,600))

print(img1.shape)
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# 2. ADIM : Karıştırılmış resim = alpha * img1 + beta * img2 --> cv2.addWeighted()
# alpha ve beta karıştırma oranları !!!!

blended = cv2.addWeighted(src1 = img1, alpha = 0.5, src2 = img2, beta = 0.5, gamma = 0)
plt.figure()
plt.imshow(blended)

# %% Görüntü Eşikleme

"""
 Görüntüleri Eşiklemenin resim üzerindeki etkisine bakacağız.
 
"""

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(img, cmap = "gray") # COLOR MAP = GRAY --> Genlik değeri 0 - 255 arası
plt.axis("off")
plt.show()

# Eşikleme (Tresholding)
# Gereksiz detaylardan bir Treshold değeri belirleyerek kurtulmak.

_, thresh_img = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY)
_, thresh_img2 = cv2.threshold(img, thresh = 60, maxval = 255, type = cv2.THRESH_BINARY_INV)
plt.figure()
plt.axis("off")
plt.imshow(thresh_img, cmap = "gray")
plt.figure()
plt.axis("off")
plt.imshow(thresh_img2, cmap = "gray")

plt.show()

# Uyarlamalı Eşik Değeri --> Adaptive Threshold
# cv2.adaptiveTreshold(resim, maxValue, Kullanılacak Yöntem Methodu, Threshold Tipi, Blocksize, C sabiti)

thresh_img3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure()
plt.imshow(thresh_img3, cmap = "gray")

"""
 * NOT : Yukarıda Kullandığımız "Adaptive Treshold" yöntemi ile resimde bir bütünlük yakaladık 
   ve bu sayede nesneleri görüntüden tamamen ayırabiliyoruz. Nesnelerin sınırları ortaya
   Çıkıyor. Bu nesne tespiti için çok önemli bir detay!!!
   
"""

# %% BULANIKLAŞTIRMA - GÜRÜLTÜ GİDERME

import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings("ignore")

# Blurring (Detayı azaltır, Gürültüyü engeller.)

img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Orjinal")
plt.show()

"""
 * Ortalama Bulanıklaştırma Yöntemi :
     
"""

dst2 = cv2.blur(img, ksize = (3,3))
plt.figure()
plt.imshow(dst2)
plt.axis("off")
plt.title("Ortalama Blur")
plt.show()

"""
 * Gaussian Blurr
 
"""

gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure()
plt.axis("off")
plt.title("Gauss Blur")
plt.imshow(gb)
plt.show()

"""
 * Median Blurr
 
 * Tuz - Biber gürültüsü giderme konusunda oldukça etkilidir!!!
 
"""

mb = cv2.medianBlur(img, ksize = 3)
plt.figure()
plt.axis("off")
plt.title("Median Blur")
plt.imshow(gb)
plt.show()

# Noise Oluşturma

def gaussianNoise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.05 # Varyans (Varience)
    sigma = var**0.5 # Standart Sapma --> Varyansın Karekökü
    
    gauss = np.random.normal(mean, sigma, (row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    
    return noisy

# İçe aktar ve Normalize (0 - 1 arası) et.
     
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255 # Normalize 0 - 1 arası
plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Orjinal")
plt.show()

# Gaussian Noise ekle ve resmi göster.

gaussianNoisyImage = gaussianNoise(img) # Orjinal resim üzerine Gaussian Noise ekler.

plt.figure()
plt.imshow(gaussianNoisyImage)
plt.axis("off")
plt.title("Gauss Noisy")
plt.show()

# Gaussian Blurr yöntemi ile Gaussian Noise Temizlemek

gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)
plt.figure()
plt.axis("off")
plt.title("With Gauss Blur")
plt.imshow(gb2)
plt.show()

# Tuz Biber Gürültüsü eklemek

def saltPepperNoise(image):
    row, col, ch = image.shape
    s_vs_p = 0.5 # Tuz biber gürültü oranı Yüzde 50 olsun.
    
    amount = 0.004
    
    noisy = np.copy(image)
    
    #salt beyaz
    num_salt = np.ceil(amount * image.size * s_vs_p) # np.ceil() --> ondalıklı sayıları tam sayıya tamamlar ve resime eklenecek gürültü sayısı belirlenir. 
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1
    
    #pepper siyah
    num_pepper = np.ceil(amount * image.size * (1-s_vs_p)) # np.ceil() --> ondalıklı sayıları tam sayıya tamamlar ve resime eklenecek gürültü sayısı belirlenir. 
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0
    
    return noisy
    

spImage = saltPepperNoise(img)
plt.figure()
plt.axis("off")
plt.title("SP Image")
plt.imshow(spImage)
plt.show()
    
# Tuz - Biber gürültüsü gidermek

"""
 * ÖNEMLİ : OpenCV Ondalıklı sayıları float32 tipinde kabul ediyor. 
            Float64 tipindeki bir değişken kullanıldığında hata veriyor!!!!
 
"""
mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize = 3) # spImage float32'ye dönüştürüldü.
plt.figure()
plt.axis("off")
plt.title("With Median Blur")
plt.imshow(mb2)
plt.show()

# %% Morfolojik İşlemler 

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Resmi içe aktarma

img = cv2.imread("datai_team.jpg", 0)
plt.figure()
plt.axis("off")
plt.title("Original Image")
plt.imshow(img, cmap = "gray")
plt.show()
    
# EROZYON (Erosion) İşlemi

"""
 *  Erozyon işlemi Ön plandaki resmin sınırlarını aşındırır!!!!
 
     Örneğin : Yazının genişliği azalır.
 
"""
kernel = np.ones((5,5), dtype = np.uint8) # Kernel --> Filtre
result = cv2.erode(img, kernel, iterations = 1) # Kernel --> Filtre
plt.figure()
plt.axis("off")
plt.title("Erozyon")
plt.imshow(result, cmap = "gray")
plt.show()

# GENİŞLEME (Dilation) İşlemi

"""
 *  Erozyon işleminin tam tersidir!!! Görüntüdeki sınırları genişletir!!!!

    Örneğin : Yazının genişliğini artırır. 
 
"""
    
result2 = cv2.dilate(img, kernel, iterations = 1)
plt.figure()
plt.axis("off")
plt.title("Dilation")
plt.imshow(result2, cmap = "gray")
plt.show()    


# WHITE NOISE (Beyaz Nokta Gürültüsü) Oluşturma

whiteNoise = np.random.randint(0,2, size = img.shape[:2]) # 0 - 1 arası rastgele random sayı oluştur.
whiteNoise = whiteNoise*255 # Normalize edilmiş (0 - 1 arası değer) whiteNoise'u 0 - 255 arası değerlere dönüştürmek.
plt.figure()
plt.axis("off")
plt.title("White Noise")
plt.imshow(whiteNoise, cmap = "gray")
plt.show() 

noise_img = whiteNoise + img
plt.figure()
plt.axis("off")
plt.title("Image with White Noise")
plt.imshow(noise_img, cmap = "gray")
plt.show()


# AÇILMA (Açınım, Opening) İşlemi --> Erosion + Dilation

"""
 * EROZYON ve GENİŞLEME işlemlerinin sırasıyla arka arkaya uygulanması işlemidir!!!!
 
 * White Noise (Beyaz Nokta Gürültüsü) azaltmak için kullanılır!!!!
 
 * EROZYON işlemi ile WHITE NOISE azaltılır ve yok eder. Ardından resim küçüldüğü için
   
   GENİŞLEME işlemi uygulanır ve tekrar orjinal resim boyutu elde edilir!!!!
 
"""
    
opening = cv2.morphologyEx(noise_img.astype(np.float32),cv2.MORPH_OPEN, kernel)
plt.figure()
plt.axis("off")
plt.title("Acilma (Opening)")
plt.imshow(opening, cmap = "gray")
plt.show()  


# BLACK NOISE (Siyah Nokta Gürültüsü) Oluşturma

blackNoise = np.random.randint(0,2, size = img.shape[:2]) # 0 - 1 arası rastgele random sayı oluştur.
blackNoise = blackNoise * -255 # Normalize edilmiş (0 - 1 arası değer) BlackNoise'u 0 - 255 arası değerlere dönüştürmek.
plt.figure()
plt.axis("off")
plt.title("Black Noise")
plt.imshow(blackNoise, cmap = "gray")
plt.show() 

black_noise_img = blackNoise + img
black_noise_img[black_noise_img <= -245] = 0
plt.figure()
plt.axis("off")
plt.title("Image with Black Noise")
plt.imshow(black_noise_img, cmap = "gray")
plt.show()


# KAPATMA (Kapanım, Closing) İşlemi

closing = cv2.morphologyEx(black_noise_img.astype(np.float32),cv2.MORPH_CLOSE, kernel)
plt.figure()
plt.axis("off")
plt.title("Kapatma (Closing)")
plt.imshow(closing, cmap = "gray")
plt.show() 


# Gradient (Gradyan) işlemi (Fark Alma) --> EDGE DETECTION

gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
plt.figure()
plt.axis("off")
plt.title("Gradient (Gradyan)")
plt.imshow(gradient, cmap = "gray")
plt.show() 

# %% GRADYANLAR

"""
 * Görüntü gradyanı, görüntüdeki yoğunluk veya renkteki yönlü bir değişikliktir.

 * Kenar Algılama için kullanılır.
 
"""

import cv2
import matplotlib.pylab as plt

img = cv2.imread("sudoku.jpg", 0)
plt.figure(), plt.axis("off"), plt.title("Original Image"), plt.imshow(img, cmap = "gray"), plt.show()

# X Eksenindeki Gradyanlar

sobelX = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 1, dy = 0, ksize = 5)
plt.figure(), plt.axis("off"), plt.title("Sobel X"), plt.imshow(sobelX, cmap = "gray"), plt.show()


# Y Eksenindeki Gradyanlar

sobelY = cv2.Sobel(img, ddepth = cv2.CV_16S, dx = 0, dy = 1, ksize = 5)
plt.figure(), plt.axis("off"), plt.title("Sobel Y"), plt.imshow(sobelY, cmap = "gray"), plt.show()


# Laplacian Gradyan --> Hem X hem de Y eksenindeki kenarları bulmak için

laplacian = cv2.Laplacian(img, ddepth = cv2.CV_16S)
plt.figure(), plt.axis("off"), plt.title("Laplacian"), plt.imshow(laplacian, cmap = "gray"), plt.show()


# %% Histogram

"""
 * Görüntüdeki renk dağılımlarını anlayabilmek açısından çok faydalı görselleştirme
   yöntemleri sunar.
   
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(),plt.title("Red - Blue"), plt.imshow(img_vis),plt.show()

print(img.shape)

# Histogram Çizdirme

img_hist = cv2.calcHist([img], channels =[0], mask = None, histSize = [256], ranges = [0,256])
print(img_hist.shape)
plt.figure(),plt.plot(img_hist) # Grafik çizdirmek için "plt.plot()" kullanılır!!

# Histogram için renk ayrımı yapma


color = ("b", "g", "r") # Tuple oluşturuldu.
plt.figure()
for i,c in enumerate(color): # Tuple içindeki verileri Index olarak i'yeString olarak ta c'ye eşitler!!! 
    hist = cv2.calcHist([img], channels =[i], mask = None, histSize = [256], ranges = [0,256])
    plt.plot(hist, color = c)                       
    

# Maskeleme işlemi ile Golden Gate resmindeki pixel değerlerinin genliklerinin dağılımlarına bakalım.
    
golden_gate = cv2.imread("goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(golden_gate_vis)
plt.show() 
    
print(golden_gate.shape)  
  
mask = np.zeros(golden_gate.shape[:2], np.uint8)  
plt.figure()   
plt.imshow(mask, cmap = "gray") # Maskeyi bütün resme uyguladığımız için resim görünmez.

mask[1500:2000, 1000:2000] = 255 # belirtilen yerlerin pixel değerleri 255 olsun. Yani Beyaz.
plt.figure(), plt.imshow(mask, cmap = "gray")    
    
# Maske Uygulanmış resim

masked_img_vis = cv2.bitwise_and(golden_gate_vis, golden_gate_vis, mask = mask)
plt.figure()   
plt.imshow(masked_img_vis, cmap = "gray")

# Orjinal resme Maske uygulamak ve Histogram grafiğini çizdirmek.
""" 
 * Resim üzerinde Red, Green, Blue değerlerinin dağılımını grafikte görüntülemek için 
   channels = [0] --> değeri ayarlanır.
   channels [0 --> RED, 1 --> GREEN, 2 --> Blue] ayarlaması yapılır.
   
"""

masked_img =  cv2.bitwise_and(golden_gate, golden_gate, mask = mask)
masked_img_hist = cv2.calcHist([golden_gate], channels =[0], mask = mask, histSize = [256], ranges = [0,256])
plt.figure()   
plt.plot(masked_img_hist)    
    

# Histogram Eşitleme Yöntemi --> Karşıtlık Artırma

"""
 * Detayları anlaşılmayan yani birbirine yakın renkteki değerleri bulunan resim için
   Kontrast arttırma işlemi yapılarak detayların ortaya çıkması sağlanır.
   
"""

img = cv2.imread("hist_equ.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray")

img_hist = cv2.calcHist([img], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(img_hist)

# Equalization (Eşitleme) İşlemi

eq_hist = cv2.equalizeHist(img) # Kontrast arttırıldı ve detaylar daha belirgin hale geldi.
plt.figure(), plt.imshow(eq_hist, cmap = "gray")


eq_img_hist = cv2.calcHist([eq_hist], channels = [0], mask = None, histSize = [256], ranges = [0,256])
plt.figure(), plt.plot(eq_img_hist)
    
# %%      
    
    
    




    