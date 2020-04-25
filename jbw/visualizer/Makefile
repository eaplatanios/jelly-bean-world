#
# List of source files
#

BIN_DIR=../../bin
JBW_VISUALIZER_CPP_SRCS=jbw_visualizer.cpp
JBW_VISUALIZER_SHADERS=$(wildcard *.spv)
JBW_VISUALIZER_DBG_OBJS=$(JBW_VISUALIZER_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
JBW_VISUALIZER_OBJS=$(JBW_VISUALIZER_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)


#
# Compile and link options
#

LIBRARY_PKG_LIBS=
PKG_LIBS=-pthread -lvulkan -lglfw
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
override CPPFLAGS_DBG += $(WARNING_FLAGS) -I. -I../deps/ -g -march=native -mtune=native -std=c++11 $(PKG_LIBS)
override CPPFLAGS += $(WARNING_FLAGS) -I. -I../deps/ -Ofast -fno-stack-protector -DNDEBUG -march=native -mtune=native -std=c++11 $(PKG_LIBS)
override LDFLAGS_DBG += -g $(LIB_PATHS)
override LDFLAGS += $(LIB_PATHS) -fwhole-program


#
# GNU Make: targets that don't build files
#

.PHONY: visualizer visualizer_dbg all debug clean distclean

#
# Make targets
#

visualizer: all
visualizer_dbg: debug

all: jbw_visualizer

debug: jbw_visualizer_dbg

-include $(JBW_VISUALIZER_OBJS:.release.o=.release.d)
-include $(JBW_VISUALIZER_DBG_OBJS:.debug.o=.debug.d)

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

$(BIN_DIR)/%.spv: %.spv
	cp $< $@

jbw_visualizer: bin $(LIBS) $(JBW_VISUALIZER_OBJS) $(JBW_VISUALIZER_SHADERS:%=$(BIN_DIR)/%)
		$(CPP) -o $(BIN_DIR)/jbw_visualizer $(CPPFLAGS) $(LDFLAGS) $(JBW_VISUALIZER_OBJS)

jbw_visualizer_dbg: bin $(LIBS) $(JBW_VISUALIZER_DBG_OBJS) $(JBW_VISUALIZER_SHADERS:%=$(BIN_DIR)/%)
		$(CPP) -o $(BIN_DIR)/jbw_visualizer_dbg $(CPPFLAGS_DBG) $(LDFLAGS_DBG) $(JBW_VISUALIZER_DBG_OBJS)

clean:
	    ${RM} -f $(BIN_DIR)/jbw_visualizer* $(JBW_VISUALIZER_SHADERS:%=$(BIN_DIR)/%) $(LIBS)

distclean:  clean
	    ${RM} -f *~
