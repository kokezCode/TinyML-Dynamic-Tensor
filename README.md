# TinyML-Dynamic-Tensor
# Gömülü Sistemler İçin Dinamik Tensör Yapısı

Bu proje, TinyML uygulamalarında RAM kullanımını optimize etmek amacıyla geliştirilmiştir. 

## Teknik Detaylar
- **Veri Yapısı:** `union` kullanılarak `float32`, `float16` ve `int8` tipleri aynı bellek adresinde yönetilir.
- **Quantization:** `tensor_quantize_to_int8_inplace` fonksiyonu ile veriler 4 kat sıkıştırılır.
- **Bellek Yönetimi:** Dinamik bellek (`malloc`/`free`) kullanımı ile kaynaklar verimli tüketilir.

## Kullanım
`main.c` dosyası üzerinden örnek bir quantization demosu çalıştırılabilir.
