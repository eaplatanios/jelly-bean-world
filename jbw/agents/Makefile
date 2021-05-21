#
# List of source files
#

BIN_DIR=../../bin
GREEDY_BLIND_AGENT_CPP_SRCS=greedy_blind_agent.cpp
GREEDY_BLIND_AGENT_DBG_OBJS=$(GREEDY_BLIND_AGENT_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
GREEDY_BLIND_AGENT_OBJS=$(GREEDY_BLIND_AGENT_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)
GREEDY_VISUAL_AGENT_CPP_SRCS=greedy_visual_agent.cpp
GREEDY_VISUAL_AGENT_DBG_OBJS=$(GREEDY_VISUAL_AGENT_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
GREEDY_VISUAL_AGENT_OBJS=$(GREEDY_VISUAL_AGENT_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)
REWARD_UPPER_BOUND_CPP_SRCS=reward_upper_bound.cpp
REWARD_UPPER_BOUND_DBG_OBJS=$(REWARD_UPPER_BOUND_CPP_SRCS:%.cpp=$(BIN_DIR)/%.debug.o)
REWARD_UPPER_BOUND_OBJS=$(REWARD_UPPER_BOUND_CPP_SRCS:%.cpp=$(BIN_DIR)/%.release.o)


#
# Compile and link options
#

CPP=g++
cc-option = $(shell $(CPP) -Werror $(1) -c -x c /dev/null -o /dev/null 2>/dev/null; echo $$?)

LIBRARY_PKG_LIBS=
PKG_LIBS=-pthread
NO_AS_NEEDED=-Wl,--no-as-needed
ifeq ($(call cc-option, $(NO_AS_NEEDED)),0)
	PKG_LIBS += $(NO_AS_NEEDED)
endif
GLIBC := $(word 2,$(shell getconf GNU_LIBC_VERSION 2>/dev/null))
ifeq "$(.SHELLSTATUS)" "0"
	GLIBC_HAS_RT := $(shell expr $(GLIBC) \>= 2.17)
	ifeq "$(GLIBC_HAS_RT)" "0"
		LIBRARY_PKG_LIBS += -lrt
		PKG_LIBS += -lrt
	endif
endif

WARNING_FLAGS=-Wall -Wpedantic
override CPPFLAGS_DBG += $(WARNING_FLAGS) -I. -I../../ -I../deps/ -g -march=native -mtune=native -std=c++11
override CPPFLAGS += $(WARNING_FLAGS) -I. -I../../ -I../deps/ -Ofast -fno-stack-protector -DNDEBUG -march=native -mtune=native -std=c++11
override LDFLAGS_DBG += -g $(LIB_PATHS) $(PKG_LIBS)
override LDFLAGS += $(LIB_PATHS) -fwhole-program $(PKG_LIBS)


#
# GNU Make: targets that don't build files
#

.PHONY: all debug clean distclean

#
# Make targets
#

agents: all
agents_dbg: debug

all: greedy_blind_agent greedy_visual_agent reward_upper_bound

debug: greedy_blind_agent_dbg greedy_visual_agent_dbg reward_upper_bound_dbg

-include $(GREEDY_BLIND_AGENT_OBJS:.release.o=.release.d)
-include $(GREEDY_BLIND_AGENT_DBG_OBJS:.debug.o=.debug.d)
-include $(GREEDY_VISUAL_AGENT_OBJS:.release.o=.release.d)
-include $(GREEDY_VISUAL_AGENT_DBG_OBJS:.debug.o=.debug.d)
-include $(REWARD_UPPER_BOUND_OBJS:.release.o=.release.d)
-include $(REWARD_UPPER_BOUND_DBG_OBJS:.debug.o=.debug.d)

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

greedy_blind_agent: bin $(LIBS) $(GREEDY_BLIND_AGENT_OBJS)
		$(CPP) -o $(BIN_DIR)/greedy_blind_agent $(GREEDY_BLIND_AGENT_OBJS) $(CPPFLAGS) $(LDFLAGS)

greedy_blind_agent_dbg: bin $(LIBS) $(GREEDY_BLIND_AGENT_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/greedy_blind_agent_dbg $(GREEDY_BLIND_AGENT_DBG_OBJS) $(CPPFLAGS_DBG) $(LDFLAGS_DBG)

greedy_visual_agent: bin $(LIBS) $(GREEDY_VISUAL_AGENT_OBJS)
		$(CPP) -o $(BIN_DIR)/greedy_visual_agent $(GREEDY_VISUAL_AGENT_OBJS) $(CPPFLAGS) $(LDFLAGS)

greedy_visual_agent_dbg: bin $(LIBS) $(GREEDY_VISUAL_AGENT_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/greedy_visual_agent_dbg $(GREEDY_VISUAL_AGENT_DBG_OBJS) $(CPPFLAGS_DBG) $(LDFLAGS_DBG)

reward_upper_bound: bin $(LIBS) $(REWARD_UPPER_BOUND_OBJS)
		$(CPP) -o $(BIN_DIR)/reward_upper_bound $(REWARD_UPPER_BOUND_OBJS) $(CPPFLAGS) $(LDFLAGS)

reward_upper_bound_dbg: bin $(LIBS) $(REWARD_UPPER_BOUND_DBG_OBJS)
		$(CPP) -o $(BIN_DIR)/reward_upper_bound_dbg $(REWARD_UPPER_BOUND_DBG_OBJS) $(CPPFLAGS_DBG) $(LDFLAGS_DBG)

clean:
	    ${RM} -f $(BIN_DIR)/greedy_blind_agent* $(BIN_DIR)/greedy_visual_agent* $(BIN_DIR)/reward_upper_bound* $(LIBS)
