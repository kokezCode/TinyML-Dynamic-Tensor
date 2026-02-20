#include <stdio.h>
#include "tensor.h"

int main(void) {
    // Örnek float giriş dizisi
    float input[] = { -1.0f, -0.5f, 0.0f, 0.25f, 0.75f, 1.0f };
    uint32_t length = (uint32_t)(sizeof(input) / sizeof(input[0]));

    Tensor t;
    tensor_init(&t);

    // 1) Float32 tensör oluştur
    int status = tensor_create_f32(&t, input, length);
    if (status != 0) {
        printf("tensor_create_f32 hata: %d\n", status);
        return 1;
    }

    size_t bytes_before = tensor_get_data_size_bytes(&t);

    // 2) Float32 -> int8 quantization (yerinde)
    status = tensor_quantize_to_int8_inplace(&t);
    if (status != 0) {
        printf("tensor_quantize_to_int8_inplace hata: %d\n", status);
        tensor_free(&t);
        return 1;
    }

    size_t bytes_after = tensor_get_data_size_bytes(&t);
    size_t bytes_saved = (bytes_before > bytes_after) ? (bytes_before - bytes_after) : 0;

    printf("Eleman sayisi     : %u\n", (unsigned)t.length);
    printf("Quantization scale: %f\n", t.scale);
    printf("Bellek (once)     : %zu byte\n", bytes_before);
    printf("Bellek (sonra)    : %zu byte\n", bytes_after);
    printf("Bellek tasarrufu  : %zu byte (yaklasik %.2fx daha az)\n",
           bytes_saved,
           bytes_after ? (double)bytes_before / (double)bytes_after : 0.0);

    // Örnek olarak quantize edilmiş int8 değerlerini de gösterelim
    if (t.type == TENSOR_TYPE_INT8 && t.data.qint8) {
        printf("Quantize edilmis int8 degerler:\n");
        for (uint32_t i = 0; i < t.length; ++i) {
            printf("%4d ", (int)t.data.qint8[i]);
        }
        printf("\n");
    }

    tensor_free(&t);

    return 0;
}