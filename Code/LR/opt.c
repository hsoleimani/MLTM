/*
 * opt.c
 *
 *  Created on: Oct 24, 2015
 *      Author: hossein
 */


#include "opt.h"

double my_f (const gsl_vector *v, void *params);
void my_df (const gsl_vector *v, void *params, gsl_vector *df);
void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df);



void optimize_w(gsl_vector * x, void * data, int n, gsl_vector * x2){

	size_t iter = 0;
	int status, j;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;

	gsl_multimin_function_fdf my_func;

	my_func.n = n;
	my_func.f = my_f;
	my_func.df = my_df;
	my_func.fdf = my_fdf;
	my_func.params = data;

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc (T, n);

	gsl_multimin_fdfminimizer_set (s, &my_func, x, 0.1, 0.01);

	do{
		iter++;
		status = gsl_multimin_fdfminimizer_iterate (s);
		//printf ("status = %s\n", gsl_strerror (status));
		if (status){
			if (iter == 1){
				for (j = 0; j < n; j++){
					gsl_vector_set(x2, j, gsl_vector_get(x, j));
				}
			}
			break;
		}

		status = gsl_multimin_test_gradient (s->gradient, 1e-3);
		if ((isnan(s->f)) || (isinf(s->f)))
			break;
		for (j = 0; j < n; j++){
			gsl_vector_set(x2, j, gsl_vector_get(s->x, j));
		}

	}while (status == GSL_CONTINUE && iter < 100);

	gsl_multimin_fdfminimizer_free (s);

}


double my_f (const gsl_vector *v, void *params)
{

	mllr_corpus * corpus = (mllr_corpus *) params;
	double f = 0.0;
	double temp, logP, maxval;
	int d, s, i, c, n, k;

	f = 0.0;

	for (k = 0; k < corpus->k; k++){
		for (n = 0; n < corpus->nterms; n++){
			corpus->w[k][n] = gsl_vector_get(v, k*corpus->nterms + n);
			corpus->grad[k][n] = 0.0;
		}
	}
	c = corpus->c;
	f = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		logP = 0.0;
		for (s = 0; s < corpus->docs[d].length; s++){
			corpus->docs[d].sents[s].py[c] = 0.0;

			maxval = -1e50;
			for (k = 0; k < corpus->k; k++){
				corpus->docs[d].sents[s].wTx[k] = 0.0;
				for (i = 0; i < corpus->docs[d].sents[s].length; i++){
					n = corpus->docs[d].sents[s].words[i];
					corpus->docs[d].sents[s].wTx[k] += corpus->w[k][n];
				}
				if (corpus->docs[d].sents[s].wTx[k] > maxval) maxval = corpus->docs[d].sents[s].wTx[k];
			}

			temp = 0.0;
			for (k = 0; k < corpus->k; k++){
				if (maxval > 0)
					corpus->docs[d].sents[s].wTx[k] = exp(corpus->docs[d].sents[s].wTx[k] - maxval);
				else
					corpus->docs[d].sents[s].wTx[k] = exp(corpus->docs[d].sents[s].wTx[k]);
				temp += corpus->docs[d].sents[s].wTx[k];
			}
			if (maxval > 0)
				temp += exp(-maxval);
			else
				temp += 1.0;
			corpus->docs[d].sents[s].py[c] = temp; //e^-maxval + \sum_k e^(wkTx - maxval)
			/*for (k = 0; k < corpus->k; k++){
				corpus->docs[d].sents[s].wTx[k] /= temp;
				if (corpus->docs[d].b[c] == 0){
					for (i = 0; i < corpus->docs[d].sents[s].length; i++){
						n = corpus->docs[d].sents[s].words[i];
						corpus->grad[k][n] -= corpus->docs[d].sents[s].wTx[k];
					}
				}
			}*/
			if (maxval > 0)
				logP -= maxval + log(corpus->docs[d].sents[s].py[c]);
			else
				logP -= log(corpus->docs[d].sents[s].py[c]);
		}

		if (corpus->docs[d].b[c] == 1)
			f += log(1-exp(logP));
		else
			f += logP;
	}



	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	mllr_corpus * corpus = (mllr_corpus *) params;
	double temp, logP, gradb1, maxval;
	int d, s, i, c, n, k;


	for (k = 0; k < corpus->k; k++){
		for (n = 0; n < corpus->nterms; n++){
			corpus->w[k][n] = gsl_vector_get(v, k*corpus->nterms + n);
			corpus->grad[k][n] = 0.0;
		}
	}
	c = corpus->c;
	//(*f) = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		logP = 0.0;
		for (s = 0; s < corpus->docs[d].length; s++){
			corpus->docs[d].sents[s].py[c] = 0.0;

			maxval = -1e50;
			for (k = 0; k < corpus->k; k++){
				corpus->docs[d].sents[s].wTx[k] = 0.0;
				for (i = 0; i < corpus->docs[d].sents[s].length; i++){
					n = corpus->docs[d].sents[s].words[i];
					corpus->docs[d].sents[s].wTx[k] += corpus->w[k][n];
				}
				if (corpus->docs[d].sents[s].wTx[k] > maxval) maxval = corpus->docs[d].sents[s].wTx[k];
			}

			temp = 0.0;
			for (k = 0; k < corpus->k; k++){
				if (maxval > 0)
					corpus->docs[d].sents[s].wTx[k] = exp(corpus->docs[d].sents[s].wTx[k] - maxval);
				else
					corpus->docs[d].sents[s].wTx[k] = exp(corpus->docs[d].sents[s].wTx[k]);
				temp += corpus->docs[d].sents[s].wTx[k];
			}
			if (maxval > 0)
				temp += exp(-maxval);
			else
				temp += 1.0;
			corpus->docs[d].sents[s].py[c] = temp; //e^-maxval + \sum_k e^(wkTx - maxval)
			for (k = 0; k < corpus->k; k++){
				corpus->docs[d].sents[s].wTx[k] /= temp;
				if (corpus->docs[d].b[c] == 0){
					for (i = 0; i < corpus->docs[d].sents[s].length; i++){
						n = corpus->docs[d].sents[s].words[i];
						corpus->grad[k][n] -= corpus->docs[d].sents[s].wTx[k];
					}
				}
			}
			if (maxval > 0)
				logP -= maxval + log(corpus->docs[d].sents[s].py[c]);
			else
				logP -= log(corpus->docs[d].sents[s].py[c]);
		}

		/*if (corpus->docs[d].b[c] == 1)
			*f += log(1-exp(logP));
		else
			*f += logP;*/

		if (corpus->docs[d].b[c] == 1){
			gradb1 = 1.0/(1-exp(logP)) - 1;
			for (s = 0; s < corpus->docs[d].length; s++){
				for (k = 0; k < corpus->k; k++){
					temp = gradb1*corpus->docs[d].sents[s].wTx[k];
					for (i = 0; i < corpus->docs[d].sents[s].length; i++){
						n = corpus->docs[d].sents[s].words[i];
						corpus->grad[k][n] += temp;
					}
				}
			}
		}

	}

	for (k = 0; k < corpus->k; k++){
		for (n = 0; n < corpus->nterms; n++){
			gsl_vector_set(df, k*corpus->nterms + n, -corpus->grad[k][n]);
		}
	}
	//*f = -(*f);

}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{


	mllr_corpus * corpus = (mllr_corpus *) params;
	double temp, logP, gradb1, maxval;
	int d, s, i, c, n, k;

	for (k = 0; k < corpus->k; k++){
		for (n = 0; n < corpus->nterms; n++){
			corpus->w[k][n] = gsl_vector_get(v, k*corpus->nterms + n);
			corpus->grad[k][n] = 0.0;
		}
	}
	c = corpus->c;
	(*f) = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		logP = 0.0;
		for (s = 0; s < corpus->docs[d].length; s++){
			corpus->docs[d].sents[s].py[c] = 0.0;

			maxval = -1e50;
			for (k = 0; k < corpus->k; k++){
				corpus->docs[d].sents[s].wTx[k] = 0.0;
				for (i = 0; i < corpus->docs[d].sents[s].length; i++){
					n = corpus->docs[d].sents[s].words[i];
					corpus->docs[d].sents[s].wTx[k] += corpus->w[k][n];
				}
				if (corpus->docs[d].sents[s].wTx[k] > maxval) maxval = corpus->docs[d].sents[s].wTx[k];
			}

			temp = 0.0;
			for (k = 0; k < corpus->k; k++){
				if (maxval > 0)
					corpus->docs[d].sents[s].wTx[k] = exp(corpus->docs[d].sents[s].wTx[k] - maxval);
				else
					corpus->docs[d].sents[s].wTx[k] = exp(corpus->docs[d].sents[s].wTx[k]);
				temp += corpus->docs[d].sents[s].wTx[k];
			}
			if (maxval > 0)
				temp += exp(-maxval);
			else
				temp += 1.0;
			corpus->docs[d].sents[s].py[c] = temp; //e^-maxval + \sum_k e^(wkTx - maxval)
			for (k = 0; k < corpus->k; k++){
				corpus->docs[d].sents[s].wTx[k] /= temp;
				if (corpus->docs[d].b[c] == 0){
					for (i = 0; i < corpus->docs[d].sents[s].length; i++){
						n = corpus->docs[d].sents[s].words[i];
						corpus->grad[k][n] -= corpus->docs[d].sents[s].wTx[k];
					}
				}
			}
			if (maxval > 0)
				logP -= maxval + log(corpus->docs[d].sents[s].py[c]);
			else
				logP -= log(corpus->docs[d].sents[s].py[c]);
		}

		if (corpus->docs[d].b[c] == 1)
			*f += log(1-exp(logP));
		else
			*f += logP;

		if (corpus->docs[d].b[c] == 1){
			gradb1 = 1.0/(1-exp(logP)) - 1;
			for (s = 0; s < corpus->docs[d].length; s++){
				for (k = 0; k < corpus->k; k++){
					temp = gradb1*corpus->docs[d].sents[s].wTx[k];
					for (i = 0; i < corpus->docs[d].sents[s].length; i++){
						n = corpus->docs[d].sents[s].words[i];
						corpus->grad[k][n] += temp;
					}
				}
			}
		}

	}

	for (k = 0; k < corpus->k; k++){
		for (n = 0; n < corpus->nterms; n++){
			gsl_vector_set(df, k*corpus->nterms + n, -corpus->grad[k][n]);
		}
	}
	*f = -(*f);
}
