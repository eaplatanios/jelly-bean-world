#
# List of source files
#

BIN_DIR=../../bin
DIFFUSION_TEST_CPP_SRCS=diffusion_test.cpp
DIFFUSION_TEST_DBG_OBJS=$(DIFFUSION_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
DIFFUSION_TEST_OBJS=$(DIFFUSION_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)
MAP_TEST_CPP_SRCS=map_test.cpp
MAP_TEST_DBG_OBJS=$(MAP_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
MAP_TEST_OBJS=$(MAP_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)
NETWORK_TEST_CPP_SRCS=network_test.cpp
NETWORK_TEST_DBG_OBJS=$(NETWORK_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
NETWORK_TEST_OBJS=$(NETWORK_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)
RENDERER_TEST_CPP_SRCS=renderer_test.cpp
RENDERER_TEST_SHADERS=renderer_test_fragment_shader.spv renderer_test_vertex_shader.spv
RENDERER_TEST_DBG_OBJS=$(RENDERER_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
RENDERER_TEST_OBJS=$(RENDERER_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)
SIMULATOR_TEST_CPP_SRCS=simulator_test.cpp
SIMULATOR_TEST_DBG_OBJS=$(SIMULATOR_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
SIMULATOR_TEST_OBJS=$(SIMULATOR_TEST_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)


#
# Compile and link options
#

LIBRARY_PKG_LIBS=
PKG_LIBS=-pthread
RENDERER_PKG_LIBS=-lvulkan -lglfw
ifeq '' '$(findstring clang,$(CPP))'
	PKG_LIBS += -Wl,--no-as-needed
endif
GLIBC := $(word 2,$(shell getconf GNU_LIBC_VERSION))
GLIBC_HAS_RT := $(shell expr $(GLIBC) \>= 2.17)
ifeq "$(GLIBC_HAS_RT)" "0"
	LIBRARY_PKG_LIBS += -lrt
	PKG_LIBS += -lrt
endif

CPP=g++
WARNING_FLAGS=-Wall -Wpedantic
override CPPFLAGS_DBG += $(WARNING_FLAGS) -I. -I../../ -I../deps/ -g -march=native -mtune=native -std=c++11 $(PKG_LIBS)
override CPPFLAGS += $(WARNING_FLAGS) -I. -I../../ -I../deps/ -Ofast -fno-stack-protector -DNDEBUG -march=native -mtune=native -std=c++11 $(PKG_LIBS)
override LDFLAGS_DBG += -g $(LIB_PATHS)
override LDFLAGS += $(LIB_PATHS) -fwhole-program


#
# GNU Make: targets that don't build files
#

.PHONY: tests tests_dbg all debug clean distclean

#
# Make targets
#

tests: all
tests_dbg: debug

all: diffusion_test map_test network_test renderer_test simulator_test

debug: diffusion_test_dbg map_test_dbg network_test_dbg renderer_test_dbg simulator_test_dbg

-include $(DIFFUSION_TEST_OBJS:.release.o=.release.d)
-include $(DIFFUSION_TEST_DBG_OBJS:.debug.o=.debug.d)
-include $(MAP_TEST_OBJS:.release.o=.release.d)
-include $(MAP_TEST_DBG_OBJS:.debug.o=.debug.d)
-include $(NETWORK_TEST_OBJS:.release.o=.release.d)
-include $(NETWORK_TEST_DBG_OBJS:.debug.o=.debug.d)
-include $(RENDERER_TEST_OBJS:.release.o=.release.d)
-include $(RENDERER_TEST_DBG_OBJS:.debug.o=.debug.d)
-include $(SIMULATOR_TEST_OBJS:.release.o=.release.d)
-include $(SIMULATOR_TEST_DBG_OBJS:.debug.o=.debug.d)

define make_dependencies
	$(1) $(2) -c $(3).$(4) -o $(BIN_DIR)/$(3).$(5).o
	$(1) -MM $(2) $(3).$(4) > $(BIN_DIR)/$(3).$(5).d
	@mv -f $(BIN_DIR)/$(3).$(5).d $(BIN_DIR)/$(3).$(5).d.tmp
	@sed -e 's|.*:|$(3).$(5).o:|' < $(BIN_DIR)/$(3).$(5).d.tmp > $(BIN_DIR)/$(3).$(5).d
	@sed -e 's/.*://' -e 's/\\$$//' < $(BIN_DIR)/$(3).$(5).d.tmp | fmt -1 | \
		sed -e 's/^ *//' -e 's/$$/:/' >> $(BIN_DIR)/$(3).$(5).d
	@rm -f $(BIN_DIR)/$(3).$(5).d.tmp
endef

$(BIN_DIR)/%.release.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS),$*,cpp,release)
$(BIN_DIR)/%.release.pic.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS),$*,cpp,release.pic)
$(BIN_DIR)/%.debug.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS_DBG),$*,cpp,debug)
$(BIN_DIR)/%.debug.pic.o: %.cpp
	$(call make_dependencies,$(CPP),$(CPPFLAGS_DBG),$*,cpp,debug.pic)

bin:
	mkdir -p $(BIN_DIR)

diffusion_test: bin $(LIBS) $(DIFFUSION_TEST_OBJS)
		$(CPP) -o $(BIN_DIR)/diffusion_test $(CPPFLAGS) $(LDFLAGS) $(DIFFUSION_TEST_OBJS)

diffusion_test_dbg: bin $(LIBS) $(DIFFUSION_TEST_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/diffusion_test_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(DIFFUSION_TEST_DBG_OBJS)

map_test: bin $(LIBS) $(MAP_TEST_OBJS)
		$(CPP) -o $(BIN_DIR)/map_test $(CPPFLAGS) $(LDFLAGS) $(MAP_TEST_OBJS)

map_test_dbg: bin $(LIBS) $(MAP_TEST_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/map_test_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(MAP_TEST_DBG_OBJS)

network_test: bin $(LIBS) $(NETWORK_TEST_OBJS)
		$(CPP) -o $(BIN_DIR)/network_test $(CPPFLAGS) $(LDFLAGS) $(NETWORK_TEST_OBJS)

network_test_dbg: bin $(LIBS) $(NETWORK_TEST_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/network_test_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(NETWORK_TEST_DBG_OBJS)

$(BIN_DIR)/%.spv: %.spv
	cp $< $@

renderer_test: bin $(LIBS) $(RENDERER_TEST_OBJS) $(RENDERER_TEST_SHADERS:%=$(BIN_DIR)/%)
		$(CPP) -o $(BIN_DIR)/renderer_test $(CPPFLAGS) $(LDFLAGS) $(RENDERER_TEST_OBJS) $(RENDERER_PKG_LIBS)

renderer_test_dbg: bin $(LIBS) $(RENDERER_TEST_DBG_OBJS) $(RENDERER_TEST_SHADERS:%=$(BIN_DIR)/%)
		$(CPP) -o $(BIN_DIR)/renderer_test_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(RENDERER_TEST_DBG_OBJS) $(RENDERER_PKG_LIBS)

simulator_test: bin $(LIBS) $(SIMULATOR_TEST_OBJS)
		$(CPP) -o $(BIN_DIR)/simulator_test $(CPPFLAGS) $(LDFLAGS) $(SIMULATOR_TEST_OBJS)

simulator_test_dbg: bin $(LIBS) $(SIMULATOR_TEST_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/simulator_test_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(SIMULATOR_TEST_DBG_OBJS)

clean:
	    ${RM} -f $(BIN_DIR)/diffusion_test* $(BIN_DIR)/map_test* $(BIN_DIR)/network_test* $(BIN_DIR)/renderer_test* $(BIN_DIR)/simulator_test* $(RENDERER_TEST_SHADERS:%=$(BIN_DIR)/%) $(LIBS)
