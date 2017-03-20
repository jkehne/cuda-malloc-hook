CFLAGS=-fPIC -O2 -I/usr/local/cuda/include
LDFLAGS=-shared -ldl

LIB=cuda_malloc_hook.so
OBJS=$(patsubst %.c,%.o,$(wildcard *.c))

$(LIB): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) $(LIB) $(OBJS)