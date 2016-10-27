/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: hossein
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "MultiLabelTM.h"
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

mltm_model* new_mltm_model(mltm_corpus* corpus, char* settings_file);
mltm_mcmc * new_mltm_mcmc(mltm_model* model);

void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name);

int compare_function(const void *a,const void *b);

void doc_mcmc_hmc(document* doc, mltm_model* model, mltm_mcmc* mcmc, int d,
		mltm_var* var, int test);
int main(int argc, char* argv[]);
void write_mltm_model(mltm_corpus * corpus, mltm_model * model, mltm_mcmc* mcmc, char * root);

void sent_mcmc(sentence* sent, mltm_model* model, mltm_mcmc* mcmc, int s, int d, int tt,
		mltm_var* var, int test);

void test_initialize(mltm_corpus* corpus, mltm_model* model, mltm_mcmc* mcmc, char* model_name);

void load_model(mltm_model* model, mltm_corpus* corpus, mltm_mcmc* mcmc, char* model_root);

mltm_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics);


void test(char* dataset, char* settings_file, char* model_name, char* dir);
void random_initialize_model(mltm_corpus* corpus, mltm_model * model, mltm_mcmc* mcmc);
double log_sum(double log_a, double log_b);

#endif /* MAIN_H_ */
