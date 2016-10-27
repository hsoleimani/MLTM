/*
 * opt.h
 *
 *  Created on: Oct 24, 2015
 *      Author: hossein
 */

#ifndef OPT_H_
#define OPT_H_

#include "SSLDA.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <gsl/gsl_multimin.h>


void optimize_w(gsl_vector * x, void * data, int n, gsl_vector * xx);
void optimize_alpha(gsl_vector * x, void * data, int n, gsl_vector * xx);
double log_sum(double log_a, double log_b);

#endif /* OPT_H_ */
