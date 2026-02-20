#include "tensor.h"

#include <stdlib.h> // malloc, free
#include <math.h>   // fabsf

void tensor_init(Tensor *t) {
    if (!t) return;
    t->type   = TENSOR_TYPE_NONE;
    t->length = 0;
    t->data.f32 = NULL;
    t->scale  = 1.0f;
}

int tensor_create_f32(Tensor *t, const float *src, uint32_t length) {
    if (!t || !src || length == 0) {
        return -1;
    }

    // Önce eski içeriği temizle
    tensor_free(t);

    float *data = (float *)malloc((size_t)length * sizeof(float));
    if (!data) {
        return -2;
    }

    for (uint32_t i = 0; i < length; ++i) {
        data[i] = src[i];
    }

    t->type     = TENSOR_TYPE_F32;
    t->length   = length;
    t->data.f32 = data;
    t->scale    = 1.0f;

    return 0;
}

void tensor_free(Tensor *t) {
    if (!t) return;

    if (t->type == TENSOR_TYPE_F32 && t->data.f32) {
        free(t->data.f32);
    } else if (t->type == TENSOR_TYPE_F16 && t->data.f16) {
        free(t->data.f16);
    } else if (t->type == TENSOR_TYPE_INT8 && t->data.qint8) {
        free(t->data.qint8);
    }

    t->type   = TENSOR_TYPE_NONE;
    t->length = 0;
    t->data.f32 = NULL;
    t->scale  = 1.0f;
}

size_t tensor_get_data_size_bytes(const Tensor *t) {
    if (!t) return 0;

    switch (t->type) {
        case TENSOR_TYPE_F32:
            return (size_t)t->length * sizeof(float);
        case TENSOR_TYPE_F16:
            return (size_t)t->length * sizeof(uint16_t);
        case TENSOR_TYPE_INT8:
            return (size_t)t->length * sizeof(int8_t);
        default:
            return 0;
    }
}

static int8_t tensor_saturate_int8(int32_t x) {
    if (x > 127)  return 127;
    if (x < -127) return -127;
    return (int8_t)x;
}

int tensor_quantize_to_int8_inplace(Tensor *t) {
    if (!t) {
        return -1;
    }
    if (t->type != TENSOR_TYPE_F32 || !t->data.f32 || t->length == 0) {
        return -1;
    }

    float *src = t->data.f32;
    uint32_t length = t->length;

    // 1) Maksimum mutlak değeri bul
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < length; ++i) {
        float a = fabsf(src[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    if (max_abs == 0.0f) {
        // Tüm elemanlar 0 -> scale tanımsız olur, 1.0 verip direkt 0 yazılabilir
        // Burada örnek olarak hata kodu döndürelim
        return -3;
    }

    float scale = 127.0f / max_abs;

    // 2) Yeni int8 buffer'ı ayır
    int8_t *qdata = (int8_t *)malloc((size_t)length * sizeof(int8_t));
    if (!qdata) {
        return -2;
    }

    // 3) Quantize et
    for (uint32_t i = 0; i < length; ++i) {
        float scaled = src[i] * scale;
        int32_t rounded = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
        qdata[i] = tensor_saturate_int8(rounded);
    }

    // 4) Eski float32 veriyi serbest bırak, union'u int8'e çevir
    free(t->data.f32);
    t->data.qint8 = qdata;
    t->type       = TENSOR_TYPE_INT8;
    t->scale      = scale;

    return 0;
}

/* ---- float32 <-> float16 yardımcı fonksiyonları ---- */
/* Basit ve gömülü ortamlara uygun, branching bazlı bir yaklaşım */

uint16_t tensor_float32_to_f16(float value) {
    // IEEE-754 single precision formatını half precision'a indirger.
    // Bu basit implementasyon, hızdan çok taşınabilirliğe odaklıdır.

    union {
        float    f;
        uint32_t u;
    } v;

    v.f = value;

    uint32_t sign = (v.u >> 31) & 0x1;
    int32_t  exp  = (int32_t)((v.u >> 23) & 0xFF) - 127 + 15; // exponent yeniden biaslanır
    uint32_t mant = (v.u >> 13) & 0x3FF;                      // 10 bit mantissa

    uint16_t result;

    if (exp <= 0) {
        // Çok küçük -> denormal veya sıfır
        if (exp < -10) {
            // Tamamen sıfırla
            result = (uint16_t)(sign << 15);
        } else {
            // Denormal: 1.mantissa'yı sağa kaydır
            uint32_t mantissa = (v.u & 0x7FFFFF) | 0x800000;
            int shift = (14 - exp);
            mantissa >>= shift;
            result = (uint16_t)((sign << 15) | (mantissa & 0x3FF));
        }
    } else if (exp >= 31) {
        // Sonsuz veya NaN
        result = (uint16_t)((sign << 15) | (0x1F << 10));
        if ((v.u & 0x7FFFFF) != 0) {
            // NaN
            result |= 0x1;
        }
    } else {
        // Normal aralık
        result = (uint16_t)((sign << 15) | ((uint16_t)exp << 10) | (uint16_t)mant);
    }

    return result;
}

float tensor_f16_to_float32(uint16_t value) {
    uint32_t sign = (value >> 15) & 0x1;
    uint32_t exp  = (value >> 10) & 0x1F;
    uint32_t mant = value & 0x3FF;

    uint32_t out_sign = sign << 31;
    uint32_t out_exp;
    uint32_t out_mant;

    if (exp == 0) {
        if (mant == 0) {
            // Sıfır
            out_exp  = 0;
            out_mant = 0;
        } else {
            // Denormal -> normalize et
            int e = -1;
            uint32_t m = mant;
            while ((m & 0x400) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3FF;
            out_exp  = (uint32_t)(127 - 15 + 1 + e) << 23;
            out_mant = m << 13;
        }
    } else if (exp == 0x1F) {
        // Sonsuz veya NaN
        out_exp  = 0xFF << 23;
        out_mant = mant ? (mant << 13) : 0;
    } else {
        // Normal
        out_exp  = (uint32_t)(exp - 15 + 127) << 23;
        out_mant = mant << 13;
    }

    union {
        uint32_t u;
        float    f;
    } v;

    v.u = out_sign | out_exp | out_mant;
    return v.f;
}