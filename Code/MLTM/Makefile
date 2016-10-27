CC = gcc -O3 -Wall -g
LDFLAGS = -lgsl -lgslcblas -lm

#GSL_INCLUDE = /usr/local/include/gsl
#GSL_LIB = /usr/local/lib
GSL_INCLUDE = /usr/global/gsl/1.16/include/
GSL_LIB = /usr/global/gsl/1.16/lib


LSOURCE = main.c 
LHEADER = MultiLabelTM.h main.h

all: $(LSOURCE) $(HEADER)
	  $(CC) -I$(GSL_INCLUDE) -L$(GSL_LIB) $(LSOURCE) $(LDFLAGS) -o MultiLabelTM

clean:
	-rm -f MultiLabelTM


