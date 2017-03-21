#include <dlfcn.h>
#include <stdint.h>
#include <unordered_map>
#include <string>
#include <iostream>

static uint32_t active_buffers = 0;
static size_t total_memory = 0;

static std::unordered_map<uintptr_t, size_t> allocs;

typedef int (*cuda_mem_alloc_v2_fp)(uintptr_t*, size_t);
typedef int (*cuda_mem_alloc_managed_fp)(uintptr_t*, size_t, unsigned int);
typedef int (*cuda_mem_alloc_pitch_v2_fp)(uintptr_t*, size_t*, size_t, size_t, unsigned int);
typedef int (*cuda_mem_free_v2_fp)(uintptr_t ptr);
typedef void *(*dlsym_fp)(void*, const char*);

extern dlsym_fp real_dlsym;
extern void *last_dlopen_handle;

static void print_alloc_message(const std::string &str) {
  std::cerr << str << " Total memory allocated: " << total_memory << ", active buffers:" << active_buffers << std::endl;
}

extern "C" {

int cuMemAlloc_v2(uintptr_t *devPtr, size_t size) {
  int ret;

  cuda_mem_alloc_v2_fp orig_cuda_mem_alloc_v2;
  orig_cuda_mem_alloc_v2 = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc_v2"));

  active_buffers++;
  total_memory += size;

  print_alloc_message("Allocation request.");

  ret = orig_cuda_mem_alloc_v2(devPtr, size);

  allocs[*devPtr] = size;

  return ret;
}

int cuMemAllocManaged (uintptr_t *dptr, size_t size, unsigned int flags) {
  int ret;

  cuda_mem_alloc_managed_fp orig_cuda_mem_alloc_managed;
  orig_cuda_mem_alloc_managed = reinterpret_cast<cuda_mem_alloc_managed_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocManaged"));

  active_buffers++;
  total_memory += size;

  print_alloc_message("Allocation request.");

  ret = orig_cuda_mem_alloc_managed(dptr, size, flags);

  allocs[*dptr] = size;

  return ret;
}

int cuMemAllocPitch_v2 (uintptr_t* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes) {
  int ret;
  size_t size;

  cuda_mem_alloc_pitch_v2_fp orig_cuda_mem_alloc_pitch_v2;
  orig_cuda_mem_alloc_pitch_v2 = reinterpret_cast<cuda_mem_alloc_pitch_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocPitch_v2"));

  ret = orig_cuda_mem_alloc_pitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);

  size = *pPitch * Height;

  active_buffers++;
  total_memory += size;

  print_alloc_message("Allocation request.");

  allocs[*dptr] = size;

  return ret;
}

int cuMemFree_v2(uintptr_t ptr) {
  cuda_mem_free_v2_fp orig_cuda_mem_free_v2;
  orig_cuda_mem_free_v2 = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemFree_v2"));

  active_buffers--;
  total_memory -= allocs[ptr];
  allocs.erase(ptr);

  print_alloc_message("Free request.");

  return orig_cuda_mem_free_v2(ptr);
}

std::unordered_map<std::string, void *> fps = {
  {"cuMemAlloc_v2", reinterpret_cast<void *>(cuMemAlloc_v2)},
  {"cuMemAllocManaged", reinterpret_cast<void *>(cuMemAllocManaged)},
  {"cuMemAllocPitch_v2", reinterpret_cast<void *>(cuMemAllocPitch_v2)},
  {"cuMemFree_v2", reinterpret_cast<void *>(cuMemFree_v2)}
};

} /* extern "C" */
