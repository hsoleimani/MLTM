
#ifndef OPT_H_
#define OPT_H_

#include "EMDD.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_multimin.h>


void optimize_w(gsl_vector * x, void * data, int n, gsl_vector * xx);


#endif /* OPT_H_ */
