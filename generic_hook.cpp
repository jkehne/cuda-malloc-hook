#include <stdint.h>
#include <sys/mman.h>
#include <string.h>
#include <stdio.h>

#include "generic_hook.hpp"

enum {
  HOOK_OFFSET = 1024,
  PAGE_SIZE   = 4096
};

static void generic_hook() {
  asm volatile(
               "push %%rdi\n\t"
               "push %%rsi\n\t"
               "push %%rdx\n\t"
               "push %%rcx\n\t"
               "push %%r8\n\t"
               "push %%r9\n\t"
               "push %%r10\n\t"
               "lea (%%rip), %%rdi\n\t"
               "and %0, %%rdi\n\t"
               "push %%rdi\n\t"
               "add %1, %%rdi\n\t"
               "mov %%rdi, %%rax\n\t"
               "add %1, %%rdi\n\t"
               "callq *(%%rax)\n\t"
               "pop %%rax\n\t"
               "pop %%r10\n\t"
               "pop %%r9\n\t"
               "pop %%r8\n\t"
               "pop %%rcx\n\t"
               "pop %%rdx\n\t"
               "pop %%rsi\n\t"
               "pop %%rdi\n\t"
               "pop %%rbp\n\t"
               "jmpq *(%%rax)"
               :
               : "i" (~0xfff), "i" (sizeof(intptr_t))
               : "rdi", "rax", "rsi", "rdx", "rcx", "r8", "r9", "r10"
	       );
}

void *make_generic_hook(const char* fn_name, void *orig_fp) {
  char *function_page = (char *)mmap(NULL, PAGE_SIZE, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  memcpy(function_page+1024, reinterpret_cast<void *>(generic_hook), HOOK_OFFSET);
  *(intptr_t *)function_page = (intptr_t)orig_fp;
  *(intptr_t *)(function_page + sizeof(intptr_t)) = (intptr_t)&puts;
  memcpy(function_page + 2*sizeof(intptr_t), fn_name, strlen(fn_name));

  mprotect(function_page, PAGE_SIZE, PROT_READ | PROT_EXEC);

  return (void *)(function_page + HOOK_OFFSET);
}
