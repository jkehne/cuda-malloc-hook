#include <iostream>

#include "stats.hpp"

void AllocStats::recordAlloc(size_t size) {
  std::lock_guard<std::mutex> guard(statsMutex);
  
  totalAllocatedBuffers++;
  totalAllocatedMemory += size;
  liveBuffers++;
  liveMemory += size;

  if (liveMemory > peakMemory)
    peakMemory = liveMemory;

  if (liveBuffers > peakBuffers)
    peakBuffers = liveBuffers;
}

void AllocStats::recordFree(size_t size) {
  std::lock_guard<std::mutex> guard(statsMutex);

  totalFreedBuffers++;
  totalFreedMemory += size;
  liveBuffers--;
  liveMemory -= size;
}

void AllocStats::print() {
  std::lock_guard<std::mutex> guard(statsMutex);

  std::cout << "Total buffers allocated: " << totalAllocatedBuffers << std::endl;
  std::cout << "Total buffers freed:     " << totalFreedBuffers << std::endl;

  if (totalAllocatedBuffers > totalFreedBuffers)
    std::cout << "WARNING: Possible memory leak!" << std::endl;

  std::cout << "Total memory allocated:  " << totalAllocatedMemory << std::endl;
  std::cout << "Total memory freed:      " << totalFreedMemory << std::endl;

  std::cout << "Peak memory:             " << peakMemory << std::endl;
  std::cout << "Peak # of buffers:       " << peakBuffers << std::endl;
}

size_t AllocStats::getCurrentMemory() {
  return liveMemory;
}

size_t AllocStats::getCurrentBuffers() {
  return liveBuffers;
}
