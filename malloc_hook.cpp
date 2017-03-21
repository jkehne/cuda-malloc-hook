#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_map>
#include <string>
#include <iostream>

static uint32_t active_buffers = 0;
static size_t total_memory = 0;

static std::unordered_map<void *, size_t> allocs;

typedef int (*cuda_malloc_fp)(void**, size_t);
typedef int (*cuda_mem_alloc_v2_fp)(unsigned int *, size_t);
typedef int (*cuda_free_fp)(void*);
typedef void *(*dlsym_fp)(void*, const char*);

extern dlsym_fp real_dlsym;
extern void *last_dlopen_handle;

static void print_alloc_message(const std::string &str) {
  std::cerr << str << " Total memory allocated: " << total_memory << ", active buffers:" << active_buffers << std::endl;
}

extern "C" {

int cudaMalloc(void **devPtr, size_t size) {
  int ret;

  cuda_malloc_fp orig_cuda_malloc;
  orig_cuda_malloc = reinterpret_cast<cuda_malloc_fp>(real_dlsym(RTLD_NEXT, "cudaMalloc"));

  active_buffers++;
  total_memory += size;

  print_alloc_message("Allocation request.");

  ret = orig_cuda_malloc(devPtr, size);

  allocs[*devPtr] = size;

  return ret;
}

int cudaFree(void *ptr) {
  cuda_free_fp orig_cuda_free;
  orig_cuda_free = reinterpret_cast<cuda_free_fp>(real_dlsym(RTLD_NEXT, "cudaFree"));

  active_buffers--;
  total_memory -= allocs[ptr];
  allocs.erase(ptr);

  print_alloc_message("Free request.");

  return orig_cuda_free(ptr);
}

int cuMemAlloc_v2(unsigned int *devPtr, size_t size) {
  int ret;

  cuda_mem_alloc_v2_fp orig_cuda_mem_alloc_v2;
  orig_cuda_mem_alloc_v2 = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc_v2"));

  active_buffers++;
  total_memory += size;

  print_alloc_message("Allocation request.");

  ret = orig_cuda_mem_alloc_v2(devPtr, size);

  allocs[reinterpret_cast<void *>(*devPtr)] = size;

  return ret;
}

std::unordered_map<std::string, void *> fps = {
  {"cuMemAlloc_v2", reinterpret_cast<void *>(cuMemAlloc_v2)}
};

} /* extern "C" */
