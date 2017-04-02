
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
		//printf("%lf\n", s->f);
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
	double temp;
	double hTs;
	int d, i, c, n, s_star;
	document* doc;
	sentence* sent;

	f = 0.0;
	hTs = 0.0;
	for (n = 0; n < corpus->nterms; n++){
		corpus->h[n] = gsl_vector_get(v, n);
		corpus->s[n] = gsl_vector_get(v, corpus->nterms + n);
		hTs += pow(corpus->h[n]*corpus->s[n], 2.0);
		corpus->grad[n] = 0.0;
		corpus->grad[n+corpus->nterms] = 0.0;
	}

	c = corpus->c;
	f = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		doc = &(corpus->docs[d]);
		s_star = doc->argmaxs[c];
		sent = &(doc->sents[s_star]);

		temp = hTs;
		for (i = 0; i < sent->length; i++){
			n = sent->words[i];
			temp += pow(corpus->s[n], 2.0)*(1-2*corpus->h[n]);
		}
		sent->py[c] = exp(-temp);

		f -= pow(sent->py[c] - doc->b[c], 2.0);
	}


	f = -f;

	return(f);
}



void my_df (const gsl_vector *v, void *params, gsl_vector *df)
{

	mllr_corpus * corpus = (mllr_corpus *) params;
	double f = 0.0;
	double temp;
	double hTs, gradterm;
	int d, i, c, n, s_star;
	document* doc;
	sentence* sent;

	f = 0.0;
	hTs = 0.0;
	for (n = 0; n < corpus->nterms; n++){
		corpus->h[n] = gsl_vector_get(v, n);
		corpus->s[n] = gsl_vector_get(v, corpus->nterms + n);
		hTs += pow(corpus->h[n]*corpus->s[n], 2.0);
		corpus->grad[n] = 0.0;
		corpus->grad[n+corpus->nterms] = 0.0;
	}

	c = corpus->c;
	f = 0.0;
	gradterm = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		doc = &(corpus->docs[d]);
		s_star = doc->argmaxs[c];
		sent = &(doc->sents[s_star]);

		temp = hTs;
		for (i = 0; i < sent->length; i++){
			n = sent->words[i];
			temp += pow(corpus->s[n], 2.0)*(1-2*corpus->h[n]);
		}
		sent->py[c] = exp(-temp);

		f -= pow(sent->py[c] - doc->b[c], 2.0);

		temp = sent->py[c]*(sent->py[c] - doc->b[c]);
		gradterm += temp;
		for (i = 0; i < sent->length; i++){
			n = sent->words[i];
			corpus->grad[n] += -2*(1)*pow(corpus->s[n], 2.0)*temp;
			corpus->grad[n+corpus->nterms] += 2*(1-2*corpus->h[n])*corpus->s[n]*temp;
		}

	}

	for (n = 0; n < corpus->nterms; n++){
		corpus->grad[n] += -2*(-corpus->h[n])*pow(corpus->s[n], 2.0)*gradterm;
		corpus->grad[n+corpus->nterms] += 2*pow(corpus->h[n],2.0)*corpus->s[n]*gradterm;
		gsl_vector_set(df, n, -corpus->grad[n]);
		gsl_vector_set(df, n+corpus->nterms, -corpus->grad[n+corpus->nterms]);
	}

	f = -f;


	//*f = -(*f);

}


void my_fdf (const gsl_vector *v, void *params, double *f, gsl_vector *df)
{


	mllr_corpus * corpus = (mllr_corpus *) params;
	double temp;
	double hTs, gradterm;
	int d, i, c, n, s_star;
	document* doc;
	sentence* sent;

	*f = 0.0;
	hTs = 0.0;
	for (n = 0; n < corpus->nterms; n++){
		corpus->h[n] = gsl_vector_get(v, n);
		corpus->s[n] = gsl_vector_get(v, corpus->nterms + n);
		hTs += pow(corpus->h[n]*corpus->s[n], 2.0);
		corpus->grad[n] = 0.0;
		corpus->grad[n+corpus->nterms] = 0.0;
	}

	c = corpus->c;
	*f = 0.0;
	gradterm = 0.0;
	for (d = 0; d < corpus->ndocs; d++){
		doc = &(corpus->docs[d]);
		s_star = doc->argmaxs[c];
		sent = &(doc->sents[s_star]);

		temp = hTs;
		for (i = 0; i < sent->length; i++){
			n = sent->words[i];
			temp += pow(corpus->s[n], 2.0)*(1-2*corpus->h[n]);
		}
		sent->py[c] = exp(-temp);

		*f -= pow(sent->py[c] - doc->b[c], 2.0);

		temp = sent->py[c]*(sent->py[c] - doc->b[c]);
		gradterm += temp;
		for (i = 0; i < sent->length; i++){
			n = sent->words[i];
			corpus->grad[n] += -2*(1)*pow(corpus->s[n], 2.0)*temp;
			corpus->grad[n+corpus->nterms] += 2*(1-2*corpus->h[n])*corpus->s[n]*temp;
		}

	}


	for (n = 0; n < corpus->nterms; n++){
		corpus->grad[n] += -2*(-corpus->h[n])*pow(corpus->s[n], 2.0)*gradterm;
		corpus->grad[n+corpus->nterms] += 2*pow(corpus->h[n],2.0)*corpus->s[n]*gradterm;
		gsl_vector_set(df, n, -corpus->grad[n]);
		gsl_vector_set(df, n+corpus->nterms, -corpus->grad[n+corpus->nterms]);
	}


	*f = -(*f);
}
