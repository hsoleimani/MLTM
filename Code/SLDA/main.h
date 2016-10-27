/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "SSLDA.h"
#include "opt.h"
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
#define sign(a) ({a < 0 ? -1.0 : 1.0;})
const gsl_rng_type * T;
gsl_rng * r;
int SENT;
double T_HMC;
int BURNIN;
double TAU;
double CONVERGED;
int MAXITER;
int NUMC;
int MCSIZE;
int NUMINIT;
int BatchSize;
double Kappa;

sslda_model* new_sslda_model(sslda_corpus* corpus, char* settings_file);
sslda_ss * new_sslda_ss(sslda_model* model);

void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name);

void mstep(sslda_corpus* corpus, sslda_model* model, sslda_ss* ss, sslda_alphaopt* aopt);
double doc_estep(document* doc, sslda_model* model, sslda_ss* ss, int d, int test);
int main(int argc, char* argv[]);
void write_sslda_model(sslda_corpus * corpus, sslda_model * model, sslda_ss* ss, char * root);

void test_initialize(sslda_corpus* corpus, sslda_model* model, sslda_ss* ss, char* model_name);

void load_model(sslda_model* model, sslda_corpus* corpus, sslda_ss* ss, char* model_root);

sslda_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics);

void test(char* dataset, char* settings_file, char* model_name, char* dir);
void random_initialize_model(sslda_corpus* corpus, sslda_model * model, sslda_ss* ss);

#endif /* MAIN_H_ */
