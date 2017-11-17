#ifndef DLSYM_HOOK_H
#define DLSYM_HOOK_H

typedef void *(*dlsym_fp)(void*, const char*);

extern dlsym_fp real_dlsym;
extern void *last_dlopen_handle;

#endif /* DLSYM_HOOK_H */
