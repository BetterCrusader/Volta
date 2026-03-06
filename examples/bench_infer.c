/*
 * bench_infer.c — benchmark volta_infer from bench_real.dll
 * Compile: clang -O2 bench_infer.c -o bench_infer.exe
 * Run:     bench_infer.exe
 */
#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

typedef void (*InferFn)(float *input, int64_t in_n, float *output, int64_t out_n);

static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
}

int main(void) {
    HMODULE dll = LoadLibraryA("bench_real.dll");
    if (!dll) { fprintf(stderr, "LoadLibrary failed: %lu\n", GetLastError()); return 1; }

    InferFn infer = (InferFn)GetProcAddress(dll, "volta_infer");
    if (!infer) { fprintf(stderr, "GetProcAddress volta_infer failed\n"); return 1; }

    /* batch=1 single-sample inference */
    {
        const int64_t in_n  = 512;
        const int64_t out_n = 1;
        const int     N   = 10000;
        float input[512]  = {0};
        float output[1]   = {0};
        for (int i = 0; i < 200; ++i) infer(input, in_n, output, out_n);
        double t0 = now_ms();
        for (int i = 0; i < N; ++i) infer(input, in_n, output, out_n);
        double t1 = now_ms();
        printf("[batch=1]  volta_infer: %d calls in %.2f ms  ->  %.2f µs/call\n",
               N, t1 - t0, (t1 - t0) * 1000.0 / N);
        printf("  output[0] = %f\n", output[0]);
    }

    /* batch=64 throughput */
    {
        const int64_t B   = 64;
        const int64_t in_n  = 512 * B;
        const int64_t out_n = B;
        const int     N   = 1000;
        float *input  = (float*)calloc(512 * B, sizeof(float));
        float *output = (float*)calloc(B, sizeof(float));
        for (int i = 0; i < 20; ++i) infer(input, in_n, output, out_n);
        double t0 = now_ms();
        for (int i = 0; i < N; ++i) infer(input, in_n, output, out_n);
        double t1 = now_ms();
        printf("[batch=64] volta_infer: %d calls in %.2f ms  ->  %.2f ms/call\n",
               N, t1 - t0, (t1 - t0) / N);
        free(input); free(output);
    }

    FreeLibrary(dll);
    return 0;
}
