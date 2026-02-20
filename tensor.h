#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h> // size_t

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TENSOR_TYPE_NONE = 0,
    TENSOR_TYPE_F32,
    TENSOR_TYPE_F16,  // float16 (uint16_t ile simüle)
    TENSOR_TYPE_INT8
} TensorType;

typedef struct {
    TensorType type;
    uint32_t   length; // eleman sayısı
    union {
        float    *f32;
        uint16_t *f16;   // float16 temsili
        int8_t   *qint8; // quantized int8
    } data;
    float scale; // int8 quantization için ölçek (örn. 127 / max_abs)
} Tensor;

/**
 * Tensor yapısını güvenli başlangıç değerlerine getirir.
 * Bellek AYIRMAZ.
 */
void tensor_init(Tensor *t);

/**
 * Verilen float diziden kopya alarak dinamik bir float32 Tensor oluşturur.
 *
 * Dönüş:
 *  0  - başarı
 * -1  - geçersiz parametre
 * -2  - bellek ayırılamadı
 */
int tensor_create_f32(Tensor *t, const float *src, uint32_t length);

/**
 * Tensor içindeki dinamik belleği serbest bırakır ve alanları sıfırlar.
 */
void tensor_free(Tensor *t);

/**
 * Tensor'u float32'den int8'e YERİNDE quantize eder.
 * - Yalnızca t->type == TENSOR_TYPE_F32 iken çağrılmalıdır.
 * - Yeni bir int8 buffer ayırır, float buffer'ı serbest bırakır.
 * - Ölçek faktörünü t->scale içinde saklar.
 *
 * Dönüş:
 *  0  - başarı
 * -1  - geçersiz parametre / tip
 * -2  - bellek ayırılamadı
 * -3  - tüm elemanlar 0 (scale hesaplanamadı)
 */
int tensor_quantize_to_int8_inplace(Tensor *t);

/**
 * Tensor verisinin anlık bellek kullanımını (byte cinsinden) döndürür.
 * Geçersiz tip için 0 döner.
 */
size_t tensor_get_data_size_bytes(const Tensor *t);

/**
 * (Opsiyonel) float32 -> float16 (uint16_t) dönüşümü.
 * IEEE-754 half precision'e yaklaşık bir dönüşüm uygular.
 */
uint16_t tensor_float32_to_f16(float value);

/**
 * (Opsiyonel) float16 (uint16_t) -> float32 dönüşümü.
 */
float tensor_f16_to_float32(uint16_t value);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H