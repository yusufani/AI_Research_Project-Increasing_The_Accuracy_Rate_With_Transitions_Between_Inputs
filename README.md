# Giriş

## Not : Bu çalışma bir araştırma çalışmasıdır. Kodlar ilerideki çalışmalara referans olması için paylaşılmıştır.

Bir yapay zeka modelinin değerlendirilme unsurlarının başında doğruluk oranı yer alır. Araştırmacıların en önemli hedefi ise bu doğruluk oranını arttıran teknikleri araştırmak ve uygulamaktır. Doğruluk oranını arttırmak için birden fazla yol seçilebilir.Bu yollar veri arttırımı, daha fazla farklı veri vermek, modelin mimarisi ve parametrelerini değiştirmek veya modelin sonuçlarını modelin kullanım amacına göre düzenlemek olabilir.

**Bu çalışmada görüntü sınıflandırmakla görevli modelin uzayını keşfederek, yeni veri eklemeden, verilerin uzayda birbirleri arasındaki geçiş matrislerinden yeni veriler üreterek başarımın arttırılması amaçlanmıştır.** İki farklı görüntü verisi arasındaki geçiş görüntüleri kısaca şöyle tarif edilebilir: İlk görüntüdeki ilgili piksel değeri ile son görüntüdeki ilgili piksel değeri arası n adet aralığa bölünür ve eşit oranda bu piksel değerleri arttırılır veya azaltılır.

Örnek olarak : 
BURAYA RESİM-1 

Bu hususta yeni eklenen verilerle iki farklı yolda deneyler yapılması amaçlanmıştır.Bu yollardan ilki yeni elde edilmiş verilerin veri kümesine dahil edilerek doğruluk oranını yükseltmek ve ikincisi elde edilen verileri kullanarak modelin test edilmesine anında müdahaleler ederek başarımın arttırılmasıdır.
# Veri Kümesi ve Mimari
Deneyler CIFAR-10 veri kümesi üzerinde yapılmıştır.CIFAR-10 32*32 boyutlu renkli 60,000 görüntü içermektedir.Bu görüntülerden 50,000 adeti eğitim kümesinde yer alırken 10,000 adeti test kümesinde yer almıştır. CIFAR-10'da toplam 10 sınıfta görüntüler sınıflandırılmıştır.Bu sınıflar uçak, araba, kuş, kedi, geyik, köpek, kurbağa, at, gemi ve kamyondan oluşur.

BURAYA ŞEKİL2 GELECEK

Deneyler CNN mimarisi üstünde denenmiştir. Seçilen [model](https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c) Python programlama dilinde, Tensorflow kütüphanesinde oluşturulmuştur. Modelde optimizasyon algoritması olarak Adam kullanılmıştır.



# Geçiş Görüntülerinin Elde Edilmesi

Modelin başarımını arttırmak için farklı kriterlerde geçiş görüntüleri oluşturabilir. Çalışmamızda yapılan kriterler aşağıda belirtilmiştir.
### 1- Sınıf Seçimi
  Görüntüleri elde ederken hedef görüntünün sınıfı farklı çeşitlerde seçilebilir.Bu seçimler; kaynak görüntü ile aynı sınıfta hedef görüntü sınıfı , rastgele hedef görüntü sınıfı veya özel olarak her kaynak görüntü sınıfı için bir hedef görüntü sınıfı olabilir. Özel olarak ayarlanmış sınıflarda her kaynak sınıfı için ilgili kaynak sınıfında başarısız olarak tahmin edilmiş 5 farklı örnek alındı.Alınan örneklerden diğer tüm sınıflarda başarılı olarak tahmin edilmiş görüntülere geçiş görüntüleri oluşturuldu. Oluşturulan bu görüntüler modele tahmin ettirildi ve geçiş görüntülerinin kaynaktan ve hedeften \%30 oranın uzaklaşmış görüntülerin ortalama kaynak resmi tahmin etme olasılığı alınarak bir karmaşıklık matrisi oluşturuldu. Bu kısımda diğer sınıflara göre daha yüksek ortalamaya sahip sınıflar bulunamamıştır. Deneyler için en yüksek oranda olanlar seçilmiştir.    
### 2- Konum Seçimi 
 Geçiş görüntülerinden alınacak örneklerin hangi noktadan alınacağı da başarımı etkileyen bir diğer unsundur. Çalışmamızda kaynak resmi başlangıç noktası olarak düşünürsek pozitif ve negatif yön olarak 2 farklı yönden örnekler alınmıştır. Pozitif yön kaynak görüntüden hedef görüntüye doğru berlirli adımlarla yaklaşmayı ifade ederken, negatif yön ise kaynak görüntüden başlayıp hedef görüntüden uzaklaşmak anlamına gelmektedir.
Seçilen yöne göre farklı uzaklıkta örnekler seçmek de başarımı etkileyen bir faktör olarak alınmıştır.Örnek olarak kaynak görüntüden pozitif yönde olarak \%50 uzaklaşmak, kaynak görüntü ile hedef görüntü arasını 100 adıma ayırırsak 50. adımdaki görüntüye ulaşmak demektir.

BURAYA ŞEKİL-3 GELECEK

Son olarak görüntü özel olarak da seçilebilir. Çalışmamızda geçiş görüntüleri elde ederken maksimumdan örnekler aldık. Maksimun ifadesi , kaynak görüntüden hedef görüntüye pozitif yönde giderken kaynak görüntünün sınıfının en yüksek olarak tahmin edildiği sınıfı ifade etmektedir.

### 3- Kriterler
Görüntüleri seçerken belirli kriterler de dikkate alınabilir.Çalışmamızda kriter olarak maksimun noktasının modelin kaynak görüntüsünün sınıfını tahmin etme oranının , modelin kaynak görüntüsünün kendi sınıfını tahmin etme oranından yüksek olan örnekler olarak aldık.


# Deneyler

### A- Eğitim Kümesine Yeni Yapay Örnekler Eklemek
Yeni yapay örnekler üretmek için geçiş görüntülerinden faydalanılmıştır. Geçiş görüntüleri üretebilmek için 3. bölümde bahsedilen  kriterlerden yararlanılmıştır. Her sınıf içi , Belirtilen kriterlerden seçilen Test kümesi üzerinde yapılan denemelerde başarım ölçütü olarak doğruluk kullanılmıştır.

Her sınıf için başarılı ve başarısız olarak 1000 görüntü seçilmiştir. Ardından başarısız sınıftan başarılı sınıfa farklı kriterler seçilerek her sınıf için 1000 örnek üretilmiştir.Toplam 10 sınıf için 10000 adet yeni yapay örnek eski görüntülerle birleştirilip karıştırılmıştır. 

Toplamda 60,000 adet görüntü içeren yeni veri kümesi için tüm modeller aynı ağırlık ve bias değerleri seçilmiştir.

BURAYA ŞEKİL-4 GELECEK

Tablo 1'deki doğruluk oranlarında modellerin başarım oranlarında düşmeler görülmektedir.Modeller arasında en başarılı olan yapay örneklerin pozitif yönde maksimun başarı gösterdiği noktadan seçilip kriterlerin uygulandığı model olmuştur.

Kriterlerin uygulanması modellerde başarının artmasını sağlamaktadır. Sınıfların seçimi konusunda rastgele ,her sınıfın kendi sınıfına gitmesi arasında kayda değer bir fark olmamaktadır. Özel sınıfların seçimi modelin başarımını düşürmektedir. Negatif yönde seçilen örnekler başarımı düşürmektedir. Yapay örneklerin en uygun seçilme noktası maksimun noktası olduğu görülmektedir.

### B- Test Başarını Yükseltmek
Modelin test anında verdiği olasılıklardan çıkarım yaparak doğruluk oranını yükseltmek başarımı yükseltmek için kullanılan yöntemlerden biridir. Çalışmamızda bir önceki başlıkta bahsettiğimiz
eğitim kümesine yeni yapay örnekler eklemek yoluyla elde ettiğimiz yeni modellerin olasılıklarını kullandık.

İki modelin birbiri ile farklı sonuçlar çıkardığı resimlerde başarım arttırılmıştır. Örnek olarak Tablo-1'deki Base-1 ve Model-5 modelleri 10000 adet veri küme içeren test kümemizde yaklaşık 6,000 adetinde doğru sınıfı seçerken , sadece Base-1 modelinin doğru tahmin ettiği 1000 adet görüntü , sadece Model-5 modelinin doğru tahmin ettiği 1000 adet görüntü vardır.Bu sonuçlardan modellerin birlikte kullanılarak daha yüksek başarımlar elde edilebileceği görülmüştür.

İki modelin sonuçlarından yeni bir veri kümesi hazırlanmıştır.Bu veri kümesinde ilk modelin 10 sınıf için verdiği tahmin oranları ve ikinci modelin 10 sınıf için verdiği tahmin oranlarına karşılık resmin gerçek sınıfı eklenmiştir.. Eğitim kümesindeki tüm görüntülerden toplamda 50,000 adet eğitim verisi elde edilmiştir.

BURAYA ŞEKİL-5 GELECEK

Elde edilen eğitim verileri .... mimarilerine girdi olarak verilip yeni modeller eğitilmiştir. Çalışmamızda XGBoost , Decision Tree , Random Forest , Naive Bayes , Support Vector Machine , K-Nearest Neighbors kullanılmıştır.Tablo-2'deki sonuçlarda da görüldüğü üzere başarım oranları gözle görülür bir artış meydana gelmiştir.


