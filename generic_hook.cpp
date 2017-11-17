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
	       //preserve anything that may be a parameter
               "push %%rax\n\t"
               "push %%rdi\n\t"
               "push %%rsi\n\t"
               "push %%rdx\n\t"
               "push %%rcx\n\t"
               "push %%r8\n\t"
               "push %%r9\n\t"
               "push %%r10\n\t"
	       "subq $0x80, %%rsp\n\t"
	       "movdqu %%xmm0, 0x0(%%rsp)\n\t"
	       "movdqu %%xmm1, 0x10(%%rsp)\n\t"
	       "movdqu %%xmm2, 0x20(%%rsp)\n\t"
	       "movdqu %%xmm3, 0x30(%%rsp)\n\t"
	       "movdqu %%xmm4, 0x40(%%rsp)\n\t"
	       "movdqu %%xmm5, 0x50(%%rsp)\n\t"
	       "movdqu %%xmm6, 0x60(%%rsp)\n\t"
	       "movdqu %%xmm7, 0x70(%%rsp)\n\t"

	       //do actual work
               "lea (%%rip), %%rdi\n\t"
               "and %0, %%rdi\n\t"
               "push %%rdi\n\t"
               "add %1, %%rdi\n\t"
               "mov %%rdi, %%r11\n\t"
               "add %1, %%rdi\n\t"
               "callq *(%%r11)\n\t"
	       "pop %%r11\n\t"

	       //restore parameters and jump to hooked function
	       "movdqu 0x0(%%rsp), %%xmm0\n\t"
	       "movdqu 0x10(%%rsp), %%xmm1\n\t"
	       "movdqu 0x20(%%rsp), %%xmm2\n\t"
	       "movdqu 0x30(%%rsp), %%xmm3\n\t"
	       "movdqu 0x40(%%rsp), %%xmm4\n\t"
	       "movdqu 0x50(%%rsp), %%xmm5\n\t"
	       "movdqu 0x60(%%rsp), %%xmm6\n\t"
	       "movdqu 0x70(%%rsp), %%xmm7\n\t"
	       "addq $0x80, %%rsp\n\t"
	       "pop %%r10\n\t"
               "pop %%r9\n\t"
               "pop %%r8\n\t"
               "pop %%rcx\n\t"
               "pop %%rdx\n\t"
               "pop %%rsi\n\t"
               "pop %%rdi\n\t"
               "pop %%rax\n\t"
               "pop %%rbp\n\t"
               "jmpq *(%%r11)"
               :
               : "i" (~0xfff), "i" (sizeof(intptr_t))
               : "rdi", "rax", "rsi", "rdx", "rcx", "r8", "r9", "r10", "r11", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"
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
