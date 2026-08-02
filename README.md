# Steam Oyun Başarı Tahmini

Bir Steam oyununun, yalnızca çıkış öncesinde bilinen bilgilerle başarılı olup olmayacağını tahmin etmeye yönelik bir makine öğrenmesi projesi. Proje, analiz notebook'u ve sonuçların incelenebildiği interaktif bir Streamlit arayüzünden oluşur.

## Veri ve problem

- **Veri seti:** [Steam Store Games](https://www.kaggle.com/datasets/nikdavis/steam-store-games), 27.075 oyun (`data/steam.csv`)
- **Hedef:** Tahmini sahip sayısı medyanın (10.000) üzerinde olan oyunlar "başarılı" kabul edilmiştir. Veride başarılı oyunların oranı yaklaşık %31'dir.
- **Kullanılan özellikler:** İngilizce desteği, multiplayer, yayıncı varlığı, platform sayısı, ücretsiz olup olmama, yaş sınırı ve çıkış mevsimi
- **Veri sızıntısı:** Yorum sayısı, oynanma süresi ve başarımlar gibi çıkış sonrası oluşan bilgiler modele dahil edilmemiştir.

## Sonuçlar

| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| Logistic Regression | 0.721 | 0.667 | 0.215 | 0.325 |
| Logistic Regression (dengeli) | 0.664 | 0.471 | 0.592 | 0.525 |
| Random Forest (dengeli) | 0.662 | 0.469 | 0.597 | 0.525 |
| XGBoost (dengeli) | 0.663 | 0.470 | 0.592 | 0.524 |

- Dengesiz veride accuracy yanıltıcıdır. Ağırlıklandırma yapılmayan model, başarılı oyunların yalnızca %21'ini bulabilmektedir. Sınıf ağırlıklandırmasıyla bu oran %59'a çıkmıştır.
- Üç model de 5-fold cross validation'da benzer sonuç vermiştir (F1 ≈ 0.52). Bu durum, sınırlayıcı etkenin model seçimi değil, çıkış öncesi bilgilerin kısıtlı olması olduğunu göstermektedir.
- En etkili özellikler oyunun ücretsiz olması, yayıncısının olması ve desteklenen platform sayısıdır. Çıkış mevsiminin etkisi düşüktür.

## Streamlit arayüzü

Arayüzde model seçilebilir ve karar eşiği değiştirilerek precision, recall, confusion matrix ve hatalı tahminlerin tahmini maliyeti incelenebilir.

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Proje yapısı

```
data/steam.csv                          # veri seti
steam_games_success_prediction.ipynb    # EDA, sızıntı kontrolü, modelleme ve değerlendirme
streamlit_app.py                        # interaktif arayüz
tests/                                  # özellik çıkarımı testleri
```

Testler `pytest` ile çalıştırılır.

## Kullanılan teknolojiler

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, pytest
