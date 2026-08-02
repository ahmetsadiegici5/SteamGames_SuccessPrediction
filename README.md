# Steam Oyun Başarı Tahmini

Bu repo, Steam’de yayınlanan oyunların “başarılı olma” ihtimalini tahmin etmeye yönelik bir makine öğrenmesi çalışmasıdır. Python ile hazırlanmış analiz/deneme kodları ve sunum için üretilen görseller bu repoda bulunur.

## Ne yapar?
- Oyun verileri üzerinden özellik çıkarımı yapmayı dener
- Farklı denemelerle başarı tahmini için modelleme/analiz yapar
- Sunum veya rapor için görsel çıktılar üretebilir


## Çalıştırma (genel)
1. Python 3 kurulu olmalı
2. (Öneri) Sanal ortam oluştur:
   - `python -m venv .venv`
   - Windows: `.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`
3. Gerekli kütüphaneleri kur (repo içinde requirements dosyası yoksa elle kurman gerekebilir)
4. Script çalıştır:
   - `python make_presentation_assets.py`

