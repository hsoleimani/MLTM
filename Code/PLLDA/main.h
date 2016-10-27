/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "PLLDA.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
//#include <gsl/gsl_rng.h>
#include <math.h>


#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
const gsl_rng_type * T;
gsl_rng * r;
double TAU;
int BURNIN;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
int BatchSize;
double Kappa;

mltm_model* new_mltm_model(mltm_corpus* corpus, char* settings_file);
mltm_mcmc * new_mltm_mcmc(mltm_model* model);

void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name);

int main(int argc, char* argv[]);
void write_mltm_model(mltm_corpus * corpus, mltm_model * model, mltm_mcmc* mcmc, char * root);

void doc_mcmc(document* doc, mltm_model* model, mltm_mcmc* mcmc, int d,
		mltm_var* var);

mltm_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics);

void random_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_mcmc* mcmc);


#endif /* MAIN_H_ */
