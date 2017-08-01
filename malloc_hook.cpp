#include <dlfcn.h>
#include <stdint.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <cstdlib>

#include "stats.hpp"
#include "globals.hpp"

typedef int (*cuda_mem_alloc_v2_fp)(uintptr_t*, size_t);
typedef int (*cuda_mem_alloc_managed_fp)(uintptr_t*, size_t, unsigned int);
typedef int (*cuda_mem_alloc_pitch_v2_fp)(uintptr_t*, size_t*, size_t, size_t, unsigned int);
typedef int (*cuda_mem_free_v2_fp)(uintptr_t ptr);
typedef void *(*dlsym_fp)(void*, const char*);

static std::unordered_map<uintptr_t, size_t> allocs;

extern dlsym_fp real_dlsym;
extern void *last_dlopen_handle;

AllocStats stats;

static void print_alloc_message(const std::string &str, uintptr_t ptr, size_t size) {
  std::cerr << str << " " << size << " Bytes allocated at address 0x" << std::hex << ptr << std::dec << ". Total memory allocated: " << stats.getCurrentMemory() << ", active buffers:" << stats.getCurrentBuffers() << std::endl;
}

static void keep_stats_alloc(AllocStats &stats, uintptr_t ptr, size_t size) {
  stats.recordAlloc(size);
  print_alloc_message("Allocation request.", ptr, size);
  allocs[ptr] = size;
}

static void keep_stats_free(AllocStats &stats, uintptr_t ptr) {
  try {
    print_alloc_message("Free request.", ptr, allocs.at(ptr));

    stats.recordFree(allocs.at(ptr));
    allocs.erase(ptr);
  }
  catch(std::out_of_range) {
    std::cout << "WARNING: Unmatched cuMemFree at address 0x" << std::hex << ptr << std::dec << std::endl;
  }
}

static destructor void exit_handler() {
  stats.print();
}

extern "C" {

int cuMemAlloc_v2(uintptr_t *devPtr, size_t size) {
  int ret;

  static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc_v2 = NULL;
  if (orig_cuda_mem_alloc_v2 == NULL)
    orig_cuda_mem_alloc_v2 = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc_v2"));

  ret = orig_cuda_mem_alloc_v2(devPtr, size);

  keep_stats_alloc(stats, *devPtr, size);

  return ret;
}

int cuMemAlloc(uintptr_t *devPtr, size_t size) {
  int ret;

  static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc = NULL;
  if (orig_cuda_mem_alloc == NULL)
    orig_cuda_mem_alloc = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc"));

  ret = orig_cuda_mem_alloc(devPtr, size);

  keep_stats_alloc(stats, *devPtr, size);

  return ret;
}

int cuMemAllocManaged (uintptr_t *dptr, size_t size, unsigned int flags) {
  int ret;

  static cuda_mem_alloc_managed_fp orig_cuda_mem_alloc_managed = NULL;
  if (orig_cuda_mem_alloc_managed == NULL)
    orig_cuda_mem_alloc_managed = reinterpret_cast<cuda_mem_alloc_managed_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocManaged"));

  ret = orig_cuda_mem_alloc_managed(dptr, size, flags);

  keep_stats_alloc(stats, *dptr, size);

  return ret;
}

int cuMemAllocPitch_v2 (uintptr_t* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes) {
  int ret;
  size_t size;

  static cuda_mem_alloc_pitch_v2_fp orig_cuda_mem_alloc_pitch_v2 = NULL;
  if (orig_cuda_mem_alloc_pitch_v2 == NULL)
    orig_cuda_mem_alloc_pitch_v2 = reinterpret_cast<cuda_mem_alloc_pitch_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocPitch_v2"));

  ret = orig_cuda_mem_alloc_pitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);

  size = *pPitch * Height;

  keep_stats_alloc(stats, *dptr, size);

  return ret;
}

int cuMemAllocPitch (uintptr_t* dptr, size_t* pPitch, size_t WidthInBytes, size_t Height, unsigned int  ElementSizeBytes) {
  int ret;
  size_t size;

  static cuda_mem_alloc_pitch_v2_fp orig_cuda_mem_alloc_pitch = NULL;
  if (orig_cuda_mem_alloc_pitch == NULL)
    orig_cuda_mem_alloc_pitch = reinterpret_cast<cuda_mem_alloc_pitch_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocPitch"));

  ret = orig_cuda_mem_alloc_pitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);

  size = *pPitch * Height;

  keep_stats_alloc(stats, *dptr, size);

  return ret;
}

int cuMemFree_v2(uintptr_t ptr) {
  static cuda_mem_free_v2_fp orig_cuda_mem_free_v2 = NULL;
  if (orig_cuda_mem_free_v2 == NULL)
    orig_cuda_mem_free_v2 = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemFree_v2"));

  keep_stats_free(stats, ptr);

  return orig_cuda_mem_free_v2(ptr);
}

int cuMemFree(uintptr_t ptr) {
  static cuda_mem_free_v2_fp orig_cuda_mem_free = NULL;
  if (orig_cuda_mem_free == NULL)
    orig_cuda_mem_free = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemFree"));

  keep_stats_free(stats, ptr);

  return orig_cuda_mem_free(ptr);
}

} /* extern "C" */

std::unordered_map<std::string, void *> fps = {
  {"cuMemAlloc_v2", reinterpret_cast<void *>(cuMemAlloc_v2)},
  {"cuMemAlloc", reinterpret_cast<void *>(cuMemAlloc)},
  {"cuMemAllocManaged", reinterpret_cast<void *>(cuMemAllocManaged)},
  {"cuMemAllocPitch_v2", reinterpret_cast<void *>(cuMemAllocPitch_v2)},
  {"cuMemAllocPitch", reinterpret_cast<void *>(cuMemAllocPitch)},
  {"cuMemFree_v2", reinterpret_cast<void *>(cuMemFree_v2)},
  {"cuMemFree", reinterpret_cast<void *>(cuMemFree)}
};

