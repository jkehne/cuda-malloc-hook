#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <unordered_map>

typedef void *(*dlsym_fp)(void*, const char*);
typedef void *(*dlopen_fp)(const char*, int);

extern std::unordered_map<std::string, void *> fps;

dlsym_fp real_dlsym = NULL;
void *last_dlopen_handle = NULL;

extern "C" {

extern void *_dl_sym(void *, const char *, void *);
void *dlsym(void *handle, const char *symbol) {
  if (real_dlsym == NULL)
    real_dlsym = reinterpret_cast<dlsym_fp>(_dl_sym(RTLD_NEXT, "dlsym", reinterpret_cast<void *>(dlsym)));

  void *ret = NULL;
  try {
    ret = fps.at(symbol);
  }
  catch(std::out_of_range) {
    ret = real_dlsym(handle, symbol);    
  }

  return ret;
}

void *dlopen(const char *filename, int flags) {
  dlopen_fp real_dlopen;

  if (real_dlsym == NULL)
    real_dlsym = reinterpret_cast<dlsym_fp>(_dl_sym(RTLD_NEXT, "dlsym", reinterpret_cast<void *>(dlsym)));

  real_dlopen = reinterpret_cast<dlopen_fp>(real_dlsym(RTLD_NEXT, "dlopen"));

  last_dlopen_handle = real_dlopen(filename, flags);

  return last_dlopen_handle;
}

}