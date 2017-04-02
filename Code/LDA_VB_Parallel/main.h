/*
 * main.h
 *
 *  Created on: Jan 3, 2014
 *      Author: Hossein Soleimani
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "vblda.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "opt.h"


//#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
const gsl_rng_type * T;
gsl_rng * rng;
double CONVERGED;
double TAU, KAPPA;
int BATCHSIZE;
int MAXITER;
int nthreads;

vblda_ss * new_vblda_ss(vblda_model* model);
vblda_model* new_vblda_model(int ntopics, int nterms, int ndocs, double alpha, double nu);
vblda_var * new_vblda_var(vblda_model* model, int maxlen);

void train_stochastic(char* dataset, char* test_dataset, int ntopics, char* start,
		char* dir, double alpha, double nu);
void train(char* dataset, int ntopics, char* start, char* dir, double alpha, double nu, char* model_name);
double doc_inference(document* doc, vblda_model* model, vblda_ss* ss,
		vblda_var* var, int d, int test, int thread_id);
void test(char* dataset, char* model_name, char* dir);

int main(int argc, char* argv[]);

void write_vblda_model(char * root, vblda_model * model, vblda_var* var);
vblda_corpus* read_data(const char* data_filename, int* maxlen);
vblda_model* load_model(char* model_root, int ndocs);
void random_initialize_model(vblda_model * model, vblda_corpus* corpus, vblda_ss* ss, vblda_var* var);
void corpus_initialize_model(vblda_model * model, vblda_corpus* corpus, vblda_ss* ss, vblda_var* var);
#endif /* MAIN_H_ */
