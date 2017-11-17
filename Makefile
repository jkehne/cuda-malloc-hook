CFLAGS=-fPIC -O2 -fno-omit-frame-pointer -std=c++0x
LDFLAGS=-shared -Wl,--no-as-needed -ldl

LIB=cuda_malloc_hook.so
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))
HEADERS=$(wildcard *.hpp)

$(LIB): $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c -o $@ $<

clean:
	$(RM) $(LIB) $(OBJS)
