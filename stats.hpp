#ifndef STATS_H
#define STATS_H

#include <inttypes.h>
#include <mutex>

class AllocStats {
private:
  uint32_t totalAllocatedBuffers;
  uint32_t totalFreedBuffers;
  uint32_t totalAllocatedMemory;
  uint32_t totalFreedMemory;
  uint32_t liveBuffers;
  uint32_t liveMemory;
  uint32_t peakMemory;
  uint32_t peakBuffers;
  std::mutex statsMutex;

public:
  void recordAlloc(size_t);
  void recordFree(size_t);
  void print();

  uint32_t getCurrentMemory();
  uint32_t getCurrentBuffers();
};

#endif /* STATS_H */
