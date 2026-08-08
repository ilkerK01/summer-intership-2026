# Oyun Satış Başarısı Tahminleme

Bir video oyununun ne kadar satacağını tahmin eden makine öğrenmesi projesi.
16.443 oyunun geçmiş verisiyle eğitildi.

Kullanıcı oyunun adını, platformunu, türünü, yayıncısını, çıkış yılını ve
Metacritic puanını giriyor. Model iki şey söylüyor:

1. **Tahmini satış:** örneğin 9,77 milyon adet
2. **Satış segmenti:** Düşük / Orta / Yüksek / Blockbuster

Yazı tabanlı bir web arayüzü var, backend Python (FastAPI).

---

## Sonuçlar

Modelin eğitimde hiç görmediği **3.289 oyun** üzerinde ölçüldü:

| Metrik | Sonuç |
|---|---|
| Segment doğruluğu | **%63,7** |
| Doğru ya da komşu segment (±1) | **%97,4** |
| Macro-F1 | 0,619 |
| Satış tahmini R² (log ölçek) | 0,597 |

Karşılaştırma için, her oyuna en yaygın segmenti söyleyen basit bir tahminci
%40,8 doğruluk alıyor.

Modeli 2011 öncesi oyunlarla eğitip 2014 sonrasında test ettiğimizde doğruluk
%57,4'e düşüyor. Bu daha zorlu ama gerçek kullanıma daha yakın bir test.

---

## Kurulum

```bash
pip install -r requirements.txt
python -m scripts.get_data
python -m src.train
uvicorn api.main:app --reload
```

Sonra tarayıcıdan <http://127.0.0.1:8000> adresini açın.

API dokümanı: <http://127.0.0.1:8000/docs>

Testler: `python -m pytest tests -v`

---

## Veri seti

[Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)
(Kaggle). VGChartz satış verisi ile Metacritic puanlarını birleştiriyor.

Proje ilk olarak klasik
[vgsales](https://www.kaggle.com/datasets/gregorut/videogamesales)
veri setiyle kurulmuştu ama sonuçlar zayıf kaldı (R² 0,39, doğruluk %59).
Sebep şuydu: bir oyunun satışını en çok belirleyen şey oyunun kalitesidir ve
o veri setinde kalite ile ilgili hiçbir bilgi yoktu. Metacritic puanı olan
veri setine geçince R² 0,60'a, doğruluk %64'e çıktı.

---

## Kullanılan özellikler

Model 11 özellik kullanıyor:

| Özellik | Açıklama |
|---|---|
| Platform | Çıkış platformu (PS4, Wii, PC gibi) |
| Genre | Tür (aksiyon, spor, nişancı gibi) |
| Publisher | Yayıncı şirket |
| Rating | ESRB yaş sınıflandırması |
| Year_of_Release | Çıkış yılı |
| Critic_Score | Metacritic eleştirmen puanı (0-100) |
| User_Score | Metacritic kullanıcı puanı (0-10) |
| Seri geçmişi (4 özellik) | Aynı serinin önceki oyunlarının satışı |

Kritik puanı ve seri geçmişi en güçlü iki sinyal.

### Seri geçmişi nasıl hesaplanıyor

Oyun adından seri çıkarılıyor: "Call of Duty: Black Ops II" → "call of duty".
Sonra o serinin **yalnızca daha önceki yıllarda** çıkmış oyunlarının satışına
bakılıyor. Aynı yıl içindeki oyunlar bile hesaba katılmıyor, çünkü tahmin
anında o bilgi henüz mevcut olmaz.

---

## Neden derin öğrenme değil

Veri tablo şeklinde ve 16 bin satır. Bu boyutta LightGBM gibi gradyan
artırmalı ağaç modelleri sinir ağlarından daha iyi sonuç veriyor. Ayrıca:

- Kategorik veriyi doğrudan işliyor, embedding katmanına gerek yok
- Eksik değerleri kendisi ele alıyor (kritik puanı oyunların yarısında yok)
- Hangi özelliğin ne kadar etkili olduğu görülebiliyor
- Eğitim 32 saniye, tahmin milisaniyeler

---

## Veri sızıntısı önlemleri

Bu tür projelerde en sık yapılan hata, hedefi dolaylı olarak içeren bir
kolonu modele vermek. Üç grup kolon bilinçli olarak dışarıda bırakıldı:

**1. Bölgesel satışlar** (`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`)

Bu dördünün toplamı zaten hedef değişkene eşit. Modele verilirse R² 0,99
çıkar ama model hiçbir şey öğrenmemiş olur.

**2. Puan sayaçları** (`Critic_Count`, `User_Count`)

Bir oyuna kaç kişinin puan verdiği, o oyun satıldıktan sonra oluşan bir
bilgi. Modele eklendiğinde R² 0,63'ten 0,72'ye çıkıyordu ama çıkış öncesi
tahmin senaryosunda bu bilgi elde olmayacağı için kullanılmadı.

**3. Oyunun ham adı**

Adın kendisi verilirse model tekil oyunları ezberleyebilir. Sadece seri
bilgisi türetiliyor.

---

## İki modelin çelişmemesi

Projede iki ayrı model var: biri satış miktarını tahmin ediyor (regresyon),
diğeri satış segmentini (sınıflandırma). Bunlar bağımsız çalıştırılsa
çelişebilirdi: biri "1,2 milyon" derken diğeri "Düşük" diyebilirdi. Bu veri
setinde ölçtüğümüz çelişki oranı %29,7 idi.

Çözüm için `src/consistency.py` yazıldı. Kısaca: regresyonun tahmini ve
belirsizliği kullanılarak segmentler üzerinde bir olasılık dağılımı
oluşturuluyor, sınıflandırıcının olasılıklarıyla birleştiriliyor ve nihai
satış tahmini seçilen segmentin içinde kalacak şekilde üretiliyor.

Sonuç: 3.289 test oyununun tamamında satış tahmini, ilan edilen segmentin
içinde. Sıfır çelişki.

---

## Proje yapısı

```
src/config.py       Tüm sabitler (segment eşikleri, model ayarları)
src/data.py         Veri yükleme ve temizleme
src/features.py     Özellik üretimi
src/consistency.py  İki modeli tutarlı biçimde birleştiren katman
src/train.py        Model eğitimi ve değerlendirme
src/predict.py      Tahmin servisi
api/main.py         FastAPI web servisi
web/index.html      Web arayüzü (tek dosya)
tests/              39 test
scripts/get_data.py Veri setini indirir
```

---

## Arayüz

Üç sekme var:

- **Tahmin:** formu doldurup tahmin alıyorsunuz. Sonuçta segment olasılıkları
  ve tahmini hangi özelliklerin etkilediği gösteriliyor.
- **Puan Etkisi:** aynı oyunun kritik puanı değiştirilerek satışın nasıl
  değiştiği grafikle gösteriliyor. Modelin öğrendiği ilginç bir örüntü var:
  Metacritic 70'in altında satış neredeyse sabit, 75'ten sonra hızla artıyor.
- **Model Performansı:** doğruluk, karışıklık matrisi, kalibrasyon ve
  karşılaştırma tabloları.

---

## Bilinen sınırlar

- Kritik puanı çıkış öncesinde bilinmiyor. Puan alanı boş bırakılabilir ama
  o zaman tahmin daha belirsiz oluyor.
- Veri 1980-2016 arasını kapsıyor. Sonraki dönemin dijital dağıtım ve
  ücretsiz oyun modelleri temsil edilmiyor.
- Satış tahmininde yüzdesel hata yüksek (medyan %59). Bunun sebebi satış
  dağılımının çok çarpık olması: 10 bin satan oyunla 80 milyon satan oyun
  aynı veri setinde. Bu yüzden segment tahmini daha güvenilir bir çıktı.
- Çıkış yılı olarak yalnızca veri setinde bulunan yıllar (1980-2016)
  girilebiliyor, dışına çıkan istekler reddediliyor.

---

## Kaynaklar

- Veri: [Kaggle, Video Game Sales with Ratings](https://www.kaggle.com/datasets/rush4ratio/video-game-sales-with-ratings)
- Model: [LightGBM](https://lightgbm.readthedocs.io/)
- Web servisi: [FastAPI](https://fastapi.tiangolo.com/)
