#include <dlfcn.h>
#include <string>
#include <stdexcept>
#include <map>

#include "globals.hpp"
#include "generic_hook.hpp"

typedef void *(*dlsym_fp)(void*, const char*);
typedef void *(*dlopen_fp)(const char*, int);

extern std::map<std::string, void *> fps;

dlsym_fp real_dlsym = NULL;
dlopen_fp real_dlopen = NULL;
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
    void *real_fp = real_dlsym(handle, symbol);
#ifdef USE_GENERIC_HOOK
    if (symbol[0] == 'c' && symbol[1] == 'u') {
      ret = make_generic_hook(symbol, real_fp);
      fps[symbol] = ret;
    } else
#endif /* USE_GENERIC_HOOK */
      ret = real_fp;
  }

  return ret;
}

void *dlopen(const char *filename, int flags) {
  if (real_dlopen == NULL)
    real_dlopen = reinterpret_cast<dlopen_fp>(_dl_sym(RTLD_NEXT, "dlopen", reinterpret_cast<void *>(dlopen)));

  last_dlopen_handle = real_dlopen(filename, flags);

  return last_dlopen_handle;
}

} /* extern "C" */
