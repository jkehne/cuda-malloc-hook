CFLAGS=-fPIC -O2 -std=c++0x
LDFLAGS=-shared -Wl,--no-as-needed -ldl

LIB=cuda_malloc_hook.so
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

$(LIB): $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) $(LIB) $(OBJS)
