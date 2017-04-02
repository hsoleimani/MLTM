#include "main.h"



int main(int argc, char* argv[])
{

	char task[40];
	char dir[400];
	char corpus_file[400];
	char label_file[400];
	char settings_file[400];
	char model_name[400];
	char init[400];
	//int num_classes, num_topics;
	//double sigma2;
	long int seed;


	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 20;
	CONVERGED = 1e-2;
	NUMINIT = 10;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

	if (argc > 1){
		if (strcmp(task, "train")==0)
		{
			strcpy(label_file,argv[4]);
			strcpy(settings_file,argv[5]);
			strcpy(init,argv[6]);
			strcpy(dir,argv[7]);
			if (strcmp(init, "load") == 0)
				strcpy(model_name, argv[8]);
			train(corpus_file, label_file, settings_file, init, dir, model_name);

			gsl_rng_free (r);
			return(0);
		}
		/*if (strcmp(task, "test")==0) //will do the prediction in python
		{
			strcpy(settings_file,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(dir,argv[6]);
			test(corpus_file, settings_file, model_name, dir);

			gsl_rng_free (r);
			return(0);
		}*/
	}
	return(0);
}


void train(char* dataset, char* lblfile, char* settings_file,
		char* start, char* dir, char* model_name)
{
	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	int iteration, nclasses, ndocs;
	double lkh, prev_lkh, conv;
	int d, n, i, c;

	mllr_corpus* corpus;
	mllr_model *model = NULL;

	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fclose(fp);

	corpus = read_data(dataset, nclasses, ndocs, 1, lblfile);
	model = new_mllr_model(corpus, settings_file);

	corpus->nterms = model->n;
	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);

	// set up the log likelihood log file
	sprintf(string, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(string, "w");

	if (strcmp(start, "random")==0){
		printf("random\n");
		random_initialize_model(corpus, model);
	}


	iteration = 0;
	sprintf(filename, "%s/%03d", dir,iteration);
	printf("%s\n",filename);
	write_mllr_model(corpus, model, filename);

	time(&t1);
	sprintf(filename, "%s/likelihood.dat", dir);
	lhood_fptr = fopen(filename, "w");

	gsl_vector * w = gsl_vector_alloc (model->n*2);
	gsl_vector * w2 = gsl_vector_alloc (model->n*2);

	int s;
	prev_lkh = -1e50;
	double temp;
	document * doc;
	sentence* sent;
	do{

		//e-step:
		lkh = 0;
		for (d = 0; d < corpus->ndocs; d++){
			doc = &(corpus->docs[d]);
			for (s = 0; s < doc->length; s++){
				sent = &(doc->sents[s]);
				for (c = 0; c < model->c; c++){
					temp = model->hTs[c];
					for (i = 0; i < sent->length; i++){
						n = sent->words[i];
						temp += pow(model->s[c][n], 2.0)*(1-2*model->h[c][n]);
					}
					sent->py[c] = exp(-temp);
					if (s == 0){
						doc->maxs[c] = sent->py[c];
						doc->argmaxs[c] = 0;
					}
					else if (sent->py[c] > doc->maxs[c]){
						doc->maxs[c] = sent->py[c];
						doc->argmaxs[c] = s;
					}
				}
			}
			for (c = 0; c < model->c; c++){
				lkh -= pow(doc->maxs[c] - doc->b[c], 2.0);
			}
		}


		//m-step
		//lkh = 0.0;
		for (c = 0; c < model->c; c++){

			// update w
			corpus->c = c;
			for (n = 0; n < model->n; n++){
				gsl_vector_set(w, n, model->h[c][n]);
				gsl_vector_set(w2, n, model->h[c][n]);

				gsl_vector_set(w, n + model->n, model->s[c][n]);
				gsl_vector_set(w2, n + model->n, model->s[c][n]);
			}
			optimize_w(w, (void*) corpus, model->n*2, w2);

			model->hTs[c] = 0.0;
			for (n = 0; n < model->n; n++){
				model->h[c][n] = gsl_vector_get(w2, n);
				model->s[c][n] = gsl_vector_get(w2, n + model->n);
				model->hTs[c] += pow(model->h[c][n]*model->s[c][n], 2.0);
			}

		}

		//if (lkh < prev_lkh)
		//	printf("oooooops, lkh decreasing!\n");

		sprintf(filename, "%s/%03d", dir,1);
		write_mllr_model(corpus, model, filename);
		time(&t2);
		conv = fabs(lkh-prev_lkh)/fabs(lkh);
		prev_lkh = lkh;
		fprintf(lhood_fptr, "%d %e %e %5ld\n", iteration, lkh, conv, (int)t2-t1);
		fflush(lhood_fptr);
		printf("***** EM ITERATION %d *****, lkh = %e, conv = %e\n", iteration, lkh, conv);

		iteration ++;

		if ((iteration > MAXITER) || (conv < CONVERGED))
			break;

	}while(1);

	fclose(lhood_fptr);

	lkh = 0;
	for (d = 0; d < corpus->ndocs; d++){
		doc = &(corpus->docs[d]);
		for (s = 0; s < doc->length; s++){
			sent = &(doc->sents[s]);
			for (c = 0; c < model->c; c++){
				temp = model->hTs[c];
				for (i = 0; i < sent->length; i++){
					n = sent->words[i];
					temp += pow(model->s[c][n], 2.0)*(1-2*model->h[c][n]);
				}
				sent->py[c] = exp(-temp);
				if (s == 0){
					doc->maxs[c] = sent->py[c];
					doc->argmaxs[c] = 0;
				}
				else if (sent->py[c] > doc->maxs[c]){
					doc->maxs[c] = sent->py[c];
					doc->argmaxs[c] = s;
				}
			}
		}
		for (c = 0; c < model->c; c++){
			doc->b[c] = doc->maxs[c];
			lkh -= pow(doc->maxs[c] - doc->b[c], 2.0);
		}
	}


	sprintf(filename, "%s/final", dir);

	write_mllr_model(corpus, model, filename);

}




mllr_model* new_mllr_model(mllr_corpus* corpus, char* settings_file)
{
	int n, c, nclasses;
	//double temp1, temp2;
	FILE* fp;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "C %d\n", &nclasses);
	fclose(fp);

	mllr_model* model = malloc(sizeof(mllr_model));
	model->c = nclasses;
	model->D = 0;
	model->n = 0;


	fp = fopen(settings_file, "r");
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &model->D);
	fscanf(fp, "N %d\n", &model->n);
	fscanf(fp, "maxiter %d\n", &MAXITER);
	fscanf(fp, "converged %lf\n", &CONVERGED);
	fclose(fp);

	model->h = malloc(sizeof(double*)*model->c);
	model->s = malloc(sizeof(double*)*model->c);
	model->hTs = malloc(sizeof(double)*model->c);
	for (c = 0; c < model->c; c++){
		model->hTs[c] = 0.0;
		model->h[c] = malloc(sizeof(double)*model->n);
		model->s[c] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			model->h[c][n] = 0.0;
			model->s[c][n] = 0.0;
		}

	}

	return(model);
}



mllr_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename)
{
	FILE *fileptr;
	int Ls, word, nd, nw, lbl;
	int Sd, s, i, c;
	mllr_corpus* corpus;

	printf("reading data from %s\n", data_filename);
	corpus = malloc(sizeof(mllr_corpus));
	//corpus->docs = 0;
	corpus->docs = (document *) malloc(sizeof(document)*(ndocs));
	corpus->nterms = 0.0;
	corpus->ndocs = 0.0;
	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "<%10d> ", &Sd) != EOF)){ // number of sentences
		//corpus->docs = (document *) realloc(corpus->docs, sizeof(document)*(nd+1));
		corpus->docs[nd].sents = (sentence *) malloc(sizeof(sentence)*Sd);
		corpus->docs[nd].length = Sd;
		corpus->docs[nd].b = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].argmaxs = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].maxs = malloc(sizeof(double)*nclasses);
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = 0;
			corpus->docs[nd].argmaxs[c] = 0;
			corpus->docs[nd].maxs[c] = 0.0;
		}
		for (s = 0; s < Sd; s++){
			fscanf(fileptr, "<%10d> ", &Ls);
			corpus->docs[nd].sents[s].words = malloc(sizeof(int)*Ls);
			corpus->docs[nd].sents[s].length = Ls;
			corpus->docs[nd].sents[s].py = malloc(sizeof(double)*nclasses);
			for (i = 0; i < Ls; i++){
				fscanf(fileptr, "%10d ", &word);
				corpus->docs[nd].sents[s].words[i] = word;
				if (word >= nw) { nw = word + 1; }
			}
		}
		nd++;
	}
	fclose(fileptr);
	corpus->grad = malloc(sizeof(double)*nw*2);
	corpus->h = malloc(sizeof(double)*nw);
	corpus->s = malloc(sizeof(double)*nw);
	for (i = 0; i < nw; i++){
		corpus->grad[i] = 0.0;
		corpus->grad[i+nw] = 0.0;
		corpus->h[i] = 0.0;
		corpus->s[i] = 0.0;
	}
	corpus->ndocs = nd;
	corpus->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);

	for (nd = 0; nd < corpus->ndocs; nd++){
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = 0;
		}
	}
	if (lblchck == 1){
		//printf("reading data from %s\n", lbl_filename);
		fileptr = fopen(lbl_filename, "r");
		for (nd = 0; nd < corpus->ndocs; nd++){
			for (c = 0; c < nclasses; c++){
				fscanf(fileptr, "%d", &lbl);
				corpus->docs[nd].b[c] = lbl;
			}
		}
		fclose(fileptr);
	}
	return(corpus);
}


void write_mllr_model(mllr_corpus * corpus, mllr_model * model, char * root)
{
	char filename[200];
	FILE* fileptr;
	int n, d, s, c;


	//h
	sprintf(filename, "%s.h", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (c = 0; c < model->c; c++){
			fprintf(fileptr, "%.10lf ", model->h[c][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//s
	sprintf(filename, "%s.s", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (c = 0; c < model->c; c++){
			fprintf(fileptr, "%.10lf ", model->s[c][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);


	//Y
	sprintf(filename, "%s.y", root);
	fileptr = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (s = 0; s < corpus->docs[d].length; s++){
			fprintf(fileptr, "<");
			for (c = 0; c < model->c; c++){
				if (c > 0)
					fprintf(fileptr, ",%2.2lf", corpus->docs[d].sents[s].py[c]);
				else
					fprintf(fileptr, "%2.2lf", corpus->docs[d].sents[s].py[c]);
			}
			fprintf(fileptr, ">");
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);
}



void random_initialize_model(mllr_corpus* corpus, mllr_model * model){

	int c, n, d, s, i;
	int ndocs;
	int* doc_list = malloc(sizeof(int)*corpus->ndocs);

	for (c = 0; c < model->c; c++){
		model->hTs[c] = 0.0;
		for (n = 0; n < model->n; n++){
			model->h[c][n] = gsl_ran_flat(r, -0.05, 0.05);
			model->s[c][n] = 0.1;
			//model->hTs[c] += pow(model->s[c][n]*model->h[c][n], 2.0);
		}
		ndocs = 0;
		for (d = 0; d < corpus->ndocs; d++){
			if (corpus->docs[d].b[c] == 1){
				doc_list[ndocs] = d;
				ndocs += 1;
			}
		}
		//choose a doc
		d = doc_list[(int) floor(gsl_rng_uniform(r)*ndocs)];
		s = (int) floor(gsl_rng_uniform(r)*corpus->docs[d].length);
		for (i = 0; i < corpus->docs[d].sents[s].length; i++){
			n = corpus->docs[d].sents[s].words[i];
			model->h[c][n] += 1.0;
		}
		model->hTs[c] = 0.0;
		for (n = 0; n < model->n; n++){
			model->hTs[c] += pow(model->s[c][n]*model->h[c][n], 2.0);
		}
	}

	free(doc_list);
}



double log_sum(double log_a, double log_b)
{
	double v;

	if (log_a < log_b){
		v = log_b+log(1 + exp(log_a-log_b));
	}else{
		v = log_a+log(1 + exp(log_b-log_a));
	}
	return(v);
}

