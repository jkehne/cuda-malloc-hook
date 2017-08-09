#include <dlfcn.h>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <string>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <atomic>

#include "stats.hpp"
#include "globals.hpp"

typedef int (*cuda_mem_alloc_v2_fp)(uintptr_t*, size_t);
typedef int (*cuda_mem_alloc_managed_fp)(uintptr_t*, size_t, unsigned int);
typedef int (*cuda_mem_alloc_pitch_v2_fp)(uintptr_t*, size_t*, size_t, size_t, unsigned int);
typedef int (*cuda_mem_free_v2_fp)(uintptr_t);
typedef int (*cuda_mem_alloc_host_v2_fp)(void **, size_t);
typedef int (*cuda_mem_free_host_fp)(void *);
typedef int (*cuda_mem_host_alloc_fp)(void **, size_t, unsigned int);
typedef int (*cuda_launch_kernel_fp) (void *, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, void *, void**, void**); 
typedef void *(*dlsym_fp)(void*, const char*);

static std::unordered_map<uintptr_t, size_t> allocs;
static std::unordered_map<void *, size_t> host_allocs;

extern dlsym_fp real_dlsym;
extern void *last_dlopen_handle;

AllocStats stats;
AllocStats host_stats;
std::atomic<std::uint64_t> launched_kernels;

static void print_alloc_message(const std::string &str, AllocStats &stats, uintptr_t ptr, size_t size) {
  std::cerr << str << " " << size << " Bytes allocated at address 0x" << std::hex << ptr << std::dec << ". Total memory allocated: " << stats.getCurrentMemory() << ", active buffers:" << stats.getCurrentBuffers() << std::endl;
}

static void keep_stats_alloc(uintptr_t ptr, size_t size) {
  stats.recordAlloc(size);
  print_alloc_message("Allocation request.", stats, ptr, size);
  allocs[ptr] = size;
}

static void keep_stats_alloc_host(void *ptr, size_t size) {
  host_stats.recordAlloc(size);
  print_alloc_message("Host allocation request.", host_stats, reinterpret_cast<uintptr_t>(ptr), size);
  host_allocs[ptr] = size;
}

static void keep_stats_free(uintptr_t ptr) {
  try {
    print_alloc_message("Free request.", stats, ptr, allocs.at(ptr));

    stats.recordFree(allocs.at(ptr));
    allocs.erase(ptr);
  }
  catch(std::out_of_range) {
    std::cout << "WARNING: Unmatched cuMemFree at address 0x" << std::hex << ptr << std::dec << std::endl;
  }
}

static void keep_stats_free_host(void *ptr) {
  try {
    print_alloc_message("Host free request.", host_stats, reinterpret_cast<uintptr_t>(ptr), host_allocs.at(ptr));

    host_stats.recordFree(host_allocs.at(ptr));
    host_allocs.erase(ptr);
  }
  catch(std::out_of_range) {
    std::cout << "WARNING: Unmatched cuMemFreeHost at address 0x" << std::hex << ptr << std::dec << std::endl;
  }
}

static destructor void exit_handler() {
  std::cout << std::endl << "==================================" << std::endl;
  std::cout << "Allocation stats for Process " << getpid() << std::endl << std::endl;
  std::cout << "VRAM:" << std::endl;
  stats.print();
  std::cout << std::endl << "SYSRAM:" << std::endl;
  host_stats.print();
  std::cout << std::endl << "Kernels launched: " << launched_kernels << std::endl;
}

extern "C" {

int cuMemAlloc_v2(uintptr_t *devPtr, size_t size) {
  int ret;

  static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc_v2 = NULL;
  if (orig_cuda_mem_alloc_v2 == NULL)
    orig_cuda_mem_alloc_v2 = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc_v2"));

  ret = orig_cuda_mem_alloc_v2(devPtr, size);

  keep_stats_alloc(*devPtr, size);

  return ret;
}

int cuMemAlloc(uintptr_t *devPtr, size_t size) {
  int ret;

  static cuda_mem_alloc_v2_fp orig_cuda_mem_alloc = NULL;
  if (orig_cuda_mem_alloc == NULL)
    orig_cuda_mem_alloc = reinterpret_cast<cuda_mem_alloc_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAlloc"));

  ret = orig_cuda_mem_alloc(devPtr, size);

  keep_stats_alloc(*devPtr, size);

  return ret;
}

int cuMemAllocManaged (uintptr_t *dptr, size_t size, unsigned int flags) {
  int ret;

  static cuda_mem_alloc_managed_fp orig_cuda_mem_alloc_managed = NULL;
  if (orig_cuda_mem_alloc_managed == NULL)
    orig_cuda_mem_alloc_managed = reinterpret_cast<cuda_mem_alloc_managed_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocManaged"));

  ret = orig_cuda_mem_alloc_managed(dptr, size, flags);

  keep_stats_alloc(*dptr, size);

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

  keep_stats_alloc(*dptr, size);

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

  keep_stats_alloc(*dptr, size);

  return ret;
}

int cuMemFree_v2(uintptr_t ptr) {
  static cuda_mem_free_v2_fp orig_cuda_mem_free_v2 = NULL;
  if (orig_cuda_mem_free_v2 == NULL)
    orig_cuda_mem_free_v2 = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemFree_v2"));

  keep_stats_free(ptr);

  return orig_cuda_mem_free_v2(ptr);
}

int cuMemFree(uintptr_t ptr) {
  static cuda_mem_free_v2_fp orig_cuda_mem_free = NULL;
  if (orig_cuda_mem_free == NULL)
    orig_cuda_mem_free = reinterpret_cast<cuda_mem_free_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemFree"));

  keep_stats_free(ptr);

  return orig_cuda_mem_free(ptr);
}

int cuMemAllocHost_v2(void **ptr, size_t size) {
  int ret;

  static cuda_mem_alloc_host_v2_fp orig_cuda_mem_alloc_host_v2 = NULL;
  if (orig_cuda_mem_alloc_host_v2 == NULL)
    orig_cuda_mem_alloc_host_v2 = reinterpret_cast<cuda_mem_alloc_host_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocHost_v2"));

  ret = orig_cuda_mem_alloc_host_v2(ptr, size);

  keep_stats_alloc_host(*ptr, size);

  return ret;
}

int cuMemAllocHost(void **ptr, size_t size) {
  int ret;

  static cuda_mem_alloc_host_v2_fp orig_cuda_mem_alloc_host = NULL;
  if (orig_cuda_mem_alloc_host == NULL)
    orig_cuda_mem_alloc_host = reinterpret_cast<cuda_mem_alloc_host_v2_fp>(real_dlsym(last_dlopen_handle, "cuMemAllocHost"));

  ret = orig_cuda_mem_alloc_host(ptr, size);

  keep_stats_alloc_host(*ptr, size);

  return ret;
}

int cuMemHostAlloc(void **ptr, size_t size, unsigned int flags) {
  int ret;

  static cuda_mem_host_alloc_fp orig_cuda_mem_host_alloc = NULL;
  if (orig_cuda_mem_host_alloc == NULL)
    orig_cuda_mem_host_alloc = reinterpret_cast<cuda_mem_host_alloc_fp>(real_dlsym(last_dlopen_handle, "cuMemHostAlloc"));

  ret = orig_cuda_mem_host_alloc(ptr, size, flags);

  keep_stats_alloc_host(*ptr, size);

  return ret;
}

int cuMemFreeHost(void *ptr) {
  static cuda_mem_free_host_fp orig_cuda_mem_free_host = NULL;
  if (orig_cuda_mem_free_host == NULL)
    orig_cuda_mem_free_host = reinterpret_cast<cuda_mem_free_host_fp>(real_dlsym(last_dlopen_handle, "cuMemFreeHost"));

  keep_stats_free_host(ptr);

  return orig_cuda_mem_free_host(ptr);
}

int cuLaunchKernel ( void * f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ, unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, void * hStream, void** kernelParams, void** extra ) {
  static cuda_launch_kernel_fp orig_cuda_launch_kernel = NULL;
  if (orig_cuda_launch_kernel == NULL)
    orig_cuda_launch_kernel = reinterpret_cast<cuda_launch_kernel_fp>(real_dlsym(last_dlopen_handle, "cuLaunchKernel"));

  launched_kernels++;

  return orig_cuda_launch_kernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

} /* extern "C" */

std::map<std::string, void *> fps = {
  {"cuMemAlloc_v2", reinterpret_cast<void *>(cuMemAlloc_v2)},
  {"cuMemAlloc", reinterpret_cast<void *>(cuMemAlloc)},
  {"cuMemAllocManaged", reinterpret_cast<void *>(cuMemAllocManaged)},
  {"cuMemAllocPitch_v2", reinterpret_cast<void *>(cuMemAllocPitch_v2)},
  {"cuMemAllocPitch", reinterpret_cast<void *>(cuMemAllocPitch)},
  {"cuMemFree_v2", reinterpret_cast<void *>(cuMemFree_v2)},
  {"cuMemFree", reinterpret_cast<void *>(cuMemFree)},
  {"cuMemAllocHost_v2", reinterpret_cast<void *>(cuMemAllocHost_v2)},
  {"cuMemAllocHost", reinterpret_cast<void *>(cuMemAllocHost)},
  {"cuMemHostAlloc", reinterpret_cast<void *>(cuMemHostAlloc)},
  {"cuMemFreeHost", reinterpret_cast<void *>(cuMemFreeHost)},
  {"cuLaunchKernel", reinterpret_cast<void *>(cuLaunchKernel)}
};

