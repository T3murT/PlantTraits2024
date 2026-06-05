# 🌿 PlantTraits2024 - Çok Kanallı Bitki Özelliği Kestirim Sistemi

Bu proje, **PlantTraits2024 - FGVC11** Kaggle yarışması kapsamında geliştirilen, bitki fotoğrafları ve çevresel/coğrafi (tabular) özellikleri birleştirerek bitkilerin 6 temel fonksiyonel özelliğini (trait) tahmin eden çok kanallı bir **Late Fusion (Geç Füzyon) Derin Öğrenme** modelidir.

Ayrıca modelin sonuçlarını simüle eden ve test etmenizi sağlayan modern, web tabanlı etkileşimli bir kullanıcı arayüzü içerir.

---

## 🚀 Proje İçeriği ve Yapısı

*   `planttraits2024_model.py`: PyTorch tabanlı model eğitimi, çapraz doğrulama (K-Fold), kayıp fonksiyonları (R2Loss + Cosine Similarity) ve inference hattını içeren ana kod dosyası.
*   `plant_traits_app.html`: Tarayıcı üzerinden doğrudan çalışabilen, bitki resmi yükleme ve iklim/toprak parametreleri ile bitki özelliklerini tahmin eden etkileşimli web arayüzü (simülatör).
*   `rapor_gorselleri.html`: Proje raporunda kullanılan akademik grafikleri (eğitim eğrileri, mimari şema, sonuçlar tablosu vb.) üreten arayüz.

---

## 📐 Model Mimarisi (Model A - Late Fusion)

Sistem, iki farklı kaynaktan gelen verileri en yüksek öznitelik seviyesinde birleştirir:

1.  **Görüntü Kolu (CNN):** iNaturalist bitki fotoğrafları `EfficientNet-B0` omurgasından geçirilerek 1280 boyutlu bir öznitelik vektörü elde edilir.
2.  **Tablo Kolu (MLP):** Konuma ait 163 adet çevresel/toprak/iklim verisi, Batch Normalization, GELU aktivasyonu ve Dropout içeren bir Yapay Sinir Ağı'ndan (MLP) geçirilerek 128 boyutlu bir vektöre indirgenir.
3.  **Füzyon & Çıkış:** Bu iki vektör birleştirilerek (1408 boyut) 6 farklı bitki özelliğini tahmin eden regresyon başlığına aktarılır.

```text
    ┌───────────────────────────┐          ┌───────────────────────────┐
    │      Bitki Görüntüsü      │          │  Ancillary Çevresel Veri  │
    │    (3 x 224 x 224 piksel) │          │     (163-boyutlu vektör)  │
    └─────────────┬─────────────┘          └─────────────┬─────────────┘
                  │                                      │
                  ▼                                      ▼
    ┌───────────────────────────┐          ┌───────────────────────────┐
    │   EfficientNet-B0 CNN     │          │    Tabular MLP Branşı     │
    │   Output: 1280-boyut      │          │    Output: 128-boyut      │
    └─────────────┬─────────────┘          └─────────────┬─────────────┘
                  │                                      │
                  └──────────────────┬───────────────────┘
                                     │ (Concatenate)
                                     ▼
                      ┌─────────────────────────────┐
                      │    Birleştirilmiş Vektör    │
                      │        (1408-boyutlu)       │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │      Regresyon Başlığı      │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Tahmin Edilen 6 Trait      │
                      │  (X4, X11, X18, X26, X50,   │
                      │            X3112)           │
                      └─────────────────────────────┘
```

---

## 📊 Deneysel Sonuçlar

| Model / Yöntem | Doğrulama $R^2$ Skoru | Ortalama RMSE | Kaggle Public $R^2$ |
| :--- | :---: | :---: | :---: |
| Önceki Model (Basit Regresyon) | -500.0000 | > 5.0000 | Başarısız (Negatif) |
| Model A (Fold 0 - Tekli Model) | 0.2453 | 0.2104 | 0.2310 |
| Model A (Fold 1 - Tekli Model) | 0.2390 | 0.2115 | 0.2290 |
| **2-Fold Ensemble (Ortalama)** | **0.2510** | **0.2012** | **0.2468** |

### Tahmin Edilen Özellikler (Traits):
*   `X4`: Gövde Yoğunluğu (SSD - mg/mm³)
*   `X11`: Özgül Yaprak Alanı (SLA - mm²/mg)
*   `X18`: Bitki Boyu (Plant Height - metre)
*   `X26`: Tohum Kuru Kütlesi (Seed Dry Mass - mg)
*   `X50`: Yaprak Azot Miktarı (g/m²)
*   `X3112`: Yaprak Alanı (Leaf Area - mm²)

---

## 💻 Kullanıcı Arayüzü (Web Arayüzü)

Proje klasöründeki `plant_traits_app.html` dosyasına çift tıklayarak uygulamayı tarayıcınızda açabilirsiniz. 

### Özellikler:
*   Bitki fotoğrafı yükleme (Sürükle-bırak desteğiyle).
*   Sıcaklık, Yağış, Rakım ve pH değerlerini ayarlayabilmeniz için dinamik sürgüler.
*   Ekolojik kurallara uygun gerçekçi tahmin simülasyonu.
*   `Chart.js` tabanlı modern bar grafik görselleştirmesi.

---

## 🛠️ Kurulum ve Çalıştırma

Model eğitimini yerel bilgisayarınızda veya Kaggle ortamında çalıştırmak için gerekli kütüphaneler:

```bash
pip install torch torchvision timm scikit-learn pandas numpy opencv-python tqdm
```

Modeli eğitmek veya test tahminleri almak için:
```bash
python planttraits2024_model.py
```
