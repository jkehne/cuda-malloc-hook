#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

uint32_t active_buffers = 0;
uint32_t total_memory = 0;

int cudaMalloc(void **devPtr, size_t size) {
  int (*orig_cuda_malloc)(void**, size_t);
  orig_cuda_malloc = dlsym(RTLD_NEXT, "cudaMalloc");

  active_buffers++;
  total_memory += size;
  fprintf(stderr, "Allocating %lu bytes. Total memory allocated: %u, active buffers: %u\n", size, total_memory, active_buffers);

  return orig_cuda_malloc(devPtr, size);
}

int cudaFree(void *ptr) {
  int (*orig_cuda_free)(void*);
  orig_cuda_free = dlsym(RTLD_NEXT, "cudaFree");

  active_buffers--;
  fprintf(stderr, "Freeing buffer. Active buffers: %u\n", active_buffers);

  return orig_cuda_free(ptr);
}
