#ifndef STATS_H
#define STATS_H

#include <inttypes.h>
#include <mutex>

class AllocStats {
private:
  uint32_t totalAllocatedBuffers;
  uint32_t totalFreedBuffers;
  size_t totalAllocatedMemory;
  size_t totalFreedMemory;
  uint32_t liveBuffers;
  size_t liveMemory;
  uint32_t peakBuffers;
  size_t peakMemory;
  std::mutex statsMutex;

public:
  void recordAlloc(size_t);
  void recordFree(size_t);
  void print();

  size_t getCurrentMemory();
  size_t getCurrentBuffers();
};

#endif /* STATS_H */
