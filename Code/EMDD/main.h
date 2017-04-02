
#ifndef MAIN_H_
#define MAIN_H_

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
//#include <gsl/gsl_rng.h>
#include <math.h>
#include "EMDD.h"
#include "opt.h"


#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
const gsl_rng_type * T;
gsl_rng * r;
double TAU;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
int BatchSize;
double Kappa;

mllr_model* new_mllr_model(mllr_corpus* corpus, char* settings_file);

void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name);

int main(int argc, char* argv[]);
void write_mllr_model(mllr_corpus * corpus, mllr_model * model, char * root);
void corpus_initialize_model(mllr_corpus* corpus, mllr_model* model);


mllr_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename);

void random_initialize_model(mllr_corpus* corpus, mllr_model * model);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
