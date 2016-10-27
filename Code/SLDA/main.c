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
	long int seed;


	seed = atoi(argv[1]);

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);
	gsl_rng_set (r, seed);

	printf("SEED = %ld\n", seed);

	MAXITER = 7000;
	BURNIN = 1000;
	CONVERGED = 1e-4;
	NUMINIT = 10;

	strcpy(task,argv[2]);
	strcpy(corpus_file,argv[3]);

	if (argc > 1)
	{
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
		if (strcmp(task, "test")==0)
		{
			strcpy(settings_file,argv[4]);
			strcpy(model_name,argv[5]);
			strcpy(dir,argv[6]);
			test(corpus_file, settings_file, model_name, dir);

			gsl_rng_free (r);
			return(0);
		}
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
	int ntopics, nclasses, ndocs;
	double temp;
	int d;

	sslda_corpus* corpus;
	sslda_model *model = NULL;
	sslda_ss *ss = NULL;

	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fscanf(fp, "N %d\n", &d);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "alpha %lf\n", &temp);
	fscanf(fp, "converged %lf\n", &CONVERGED);
	fclose(fp);

	corpus = read_data(dataset, nclasses, ndocs, 1, lblfile, ntopics);
	model = new_sslda_model(corpus, settings_file);
	ss = new_sslda_ss(model);


	corpus->nterms = model->n;
	int iteration = 0;

	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);


	if (strcmp(start, "random")==0){
		printf("random\n");
		random_initialize_model(corpus, model, ss);

		sprintf(filename, "%s/likelihood.dat", dir);
		lhood_fptr = fopen(filename, "w");
	}
	else if (strcmp(start, "load")==0){
		printf("load\n");
		random_initialize_model(corpus, model, ss); //just to init phi, gamma

		load_model(model, corpus, ss, model_name);

		sprintf(filename, "%s/likelihood.dat", dir);
		lhood_fptr = fopen(filename, "a");
	}

	sslda_alphaopt* aopt = malloc(sizeof(sslda_alphaopt));
	aopt->m = model->m;
	aopt->d = model->D;

	//sprintf(filename, "%s/%03d", dir,(int)iteration);
	//printf("%s\n",filename);
	//write_sslda_model(corpus, model, ss, filename);

	time(&t1);


	double lkh, prev_lkh, conv;

	prev_lkh = -1e100;
	lkh = 0.0;
	model->logalpha = log(model->alpha);
	do{

		ss->alpha = 0.0;
		lkh = 0.0;
		for (d = 0; d < corpus->ndocs; d++){

			lkh += doc_estep(&(corpus->docs[d]), model, ss, d, 0);

		}
		printf("done with estep\n");
		mstep(corpus, model, ss, aopt);

		conv = fabs(lkh - prev_lkh)/fabs(prev_lkh);

		sprintf(filename, "%s/%03d", dir,1);
		write_sslda_model(corpus, model, ss, filename);
		time(&t2);
		fprintf(lhood_fptr, "%d %e %5ld %lf\n", iteration, lkh, (int)t2-t1, conv);
		fflush(lhood_fptr);
		printf("***** VI ITERATION %d \n", iteration);

		if ((iteration > MAXITER) || (conv < CONVERGED))
			break;
		iteration += 1;
		prev_lkh = lkh;

	}while(1);

	fclose(lhood_fptr);

	//do prediction
	double* b = malloc(sizeof(double)*model->c);
	sprintf(filename, "%s/final.b", dir);
	fp = fopen(filename, "w");
	int c, j;
	double nd;
	for (d = 0; d < corpus->ndocs; d++){

		lkh += doc_estep(&(corpus->docs[d]), model, ss, d, 0);

		nd = (double) corpus->docs[d].total;
		for (c = 0; c < model->c; c++){
			temp = 0.0;
			for (j = 0; j < model->m; j++){
				temp += model->w[j][c]*corpus->docs[d].sumphi[j]/nd;
			}

			if (temp < 0){
				b[c] = exp(temp)/(1+exp(temp));
			}else{
				b[c] = 1.0/(1+exp(-temp));
			}
			fprintf(fp, "%5.10lf ", b[c]);

		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	sprintf(filename, "%s/final", dir);

	write_sslda_model(corpus, model, ss, filename);

}

void mstep(sslda_corpus* corpus, sslda_model* model, sslda_ss* ss, sslda_alphaopt* aopt){

	int j, n, c, d, i;
	double nd, cnt;
	document* doc;
	gsl_vector * w = gsl_vector_alloc (model->m);
	gsl_vector * w2 = gsl_vector_alloc (model->m);
	gsl_vector * alpha = gsl_vector_alloc (1);
	gsl_vector * alpha2 = gsl_vector_alloc (1);

	for (j = 0; j < model->m; j++){
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = ss->t[j][n]/ss->sumt[j];
			if (model->beta[j][n] == 0)
				model->beta[j][n] = EPS;
			model->logbeta[j][n] = log(model->beta[j][n]);

			ss->t[j][n] = 0.0;
		}
		ss->sumt[j] = 0.0;
	}

	for (c = 0; c < model->c; c++){

		corpus->c = c;
		corpus->m = model->m;

		for (j = 0; j < model->m; j++){
			gsl_vector_set(w, j, model->w[j][c]);
			gsl_vector_set(w2, j, model->w[j][c]);
		}
		optimize_w(w, (void*) corpus, model->m, w2);

		for (j = 0; j < model->m; j++){
			model->w[j][c] = gsl_vector_get(w2, j);
		}

	}

	// update xi
	double h_i;
	for (d = 0; d < model->D; d++){
		if (corpus->docs[d].b[0] == -1) //skip unlabeled docs
			continue;

		doc = &(corpus->docs[d]);

		nd = (double) doc->total;
		for (c = 0; c < model->c; c++){
			doc->xi[c] = 0.0;
		}
		for (i = 0; i < doc->length; i++){
			n = doc->words[i];
			cnt = (double) doc->counts[i];

			for (c = 0; c < model->c; c++){
				h_i = 0.0;
				for (j = 0; j < model->m; j++){
					if (doc->phi[i][j] == 0)
						continue;
					h_i += doc->phi[i][j]*exp(cnt*model->w[j][c]/nd);
				}
				doc->xi[c] += log(h_i);
			}
		}
	}

	//update alpha
	gsl_vector_set(alpha, 0, model->logalpha);
	gsl_vector_set(alpha2, 0, model->logalpha);
	aopt->m = model->m;
	aopt->d = model->D;
	aopt->ss = ss->alpha;
	optimize_alpha(alpha, (void*) aopt, 1, alpha2);
	model->logalpha = gsl_vector_get(alpha2, 0);
	model->alpha = exp(model->logalpha);
	ss->alpha = 0.0;
	printf("%lf\n",model->alpha);

	gsl_vector_free(w);
	gsl_vector_free(w2);
	gsl_vector_free(alpha);
	gsl_vector_free(alpha2);
}

double doc_estep(document* doc, sslda_model* model, sslda_ss* ss, int d, int test){

	int i, j, n, c, iter;
	double cnt, normsum, temp, doclkh;
	double conv, oldlkh, h;
	double nd = (double) doc->total;
	//assuming if a doc is unlabeled, all classes in that doc are unlabeled
	if (doc->b[0] == -1){ //simple LDA
		oldlkh = -1e100;
		iter  = 0;
		do{
			doclkh = 0.0;
			for (i = 0; i < doc->length; i++){
				n = doc->words[i];
				cnt = (double) doc->counts[i];

				for (j = 0; j < model->m; j++){
					ss->oldphi[j] = doc->phi[i][j];

					doc->phi[i][j] = gsl_sf_psi(doc->gamma[j]) + model->logbeta[j][n];
					if (j > 0)
						normsum = log_sum(normsum, doc->phi[i][j]);
					else
						normsum = doc->phi[i][j];
				}
				for (j = 0; j < model->m; j++){
					doc->phi[i][j] = exp(doc->phi[i][j] - normsum);
					temp = cnt*(doc->phi[i][j] - ss->oldphi[j]);
					doc->gamma[j] += temp;
					doc->sumphi[j] += temp;
					doc->sumgamma += temp;

					if (doc->phi[i][j] > 0){
						doclkh += cnt*doc->phi[i][j]*(model->logbeta[j][n] - log(doc->phi[i][j]));
					}
				}
			}
			doclkh -= lgamma(doc->sumgamma);
			for (j = 0; j < model->m; j++){
				doclkh += lgamma(doc->gamma[j]);
			}
			conv = fabs(doclkh - oldlkh)/fabs(oldlkh);
			if ((doclkh < oldlkh) && (conv > 1e-5)){
				printf("ooops 1, %lf %lf, %lf\n", doclkh, oldlkh, conv);
			}
			if ((iter > model->T) || (conv < CONVERGED))
				break;

			oldlkh = doclkh;
			iter += 1;
		}while(1);
	}
	else{ //supervised case

		oldlkh = -1e100;
		iter  = 0;
		for (c = 0; c < model->c; c++){
			ss->xi[c] = 1.0 + exp(doc->xi[c]);
		}
		do{
			doclkh = 0.0;
			for (i = 0; i < doc->length; i++){
				n = doc->words[i];
				cnt = (double) doc->counts[i];

				for (c = 0; c < model->c; c++){
					h = 0.0;
					for (j = 0; j < model->m; j++){
						if (doc->phi[i][j] == 0)
							continue;
						h += doc->phi[i][j]*exp(cnt*model->w[j][c]/nd);
					}
					ss->hi[c] = exp(doc->xi[c] - log(h));
					ss->oldxi[c] = doc->xi[c] - log(h);
				}

				for (j = 0; j < model->m; j++){
					ss->oldphi[j] = doc->phi[i][j];

					doc->phi[i][j] = gsl_sf_psi(doc->gamma[j]) + model->logbeta[j][n];
					for (c = 0; c < model->c; c++){
						doc->phi[i][j] -= ss->hi[c]*exp(cnt*model->w[j][c]/nd)/(ss->xi[c]*cnt);
						if (doc->b[c] == 1)
							doc->phi[i][j] += model->w[j][c]/nd;
					}

					if (j > 0)
						normsum = log_sum(normsum, doc->phi[i][j]);
					else
						normsum = doc->phi[i][j];
				}
				for (j = 0; j < model->m; j++){
					doc->phi[i][j] = exp(doc->phi[i][j] - normsum);
					temp = cnt*(doc->phi[i][j] - ss->oldphi[j]);
					doc->gamma[j] += temp;
					doc->sumphi[j] += temp;
					doc->sumgamma += temp;

					if (doc->phi[i][j] > 0){
						doclkh += cnt*doc->phi[i][j]*(model->logbeta[j][n] - log(doc->phi[i][j]));
					}
				}

				for (c = 0; c < model->c; c++){
					h = 0.0;
					for (j = 0; j < model->m; j++){
						if (doc->phi[i][j] == 0)
							continue;
						h += doc->phi[i][j]*exp(cnt*model->w[j][c]/nd);
					}
					doc->xi[c] = ss->oldxi[c] + log(h);
					ss->xi[c] = 1.0 + exp(doc->xi[c]);
				}

			}

			doclkh -= lgamma(doc->sumgamma);
			for (j = 0; j < model->m; j++){
				doclkh += lgamma(doc->gamma[j]);
				for (c = 0; c < model->c; c++){
					if (doc->b[c] == 1){
						doclkh += model->w[j][c]*doc->sumphi[j]/nd;
					}
				}
			}
			for (c = 0; c < model->c; c++){
				doclkh -= log(ss->xi[c]);
			}
			conv = fabs(doclkh - oldlkh)/fabs(oldlkh);
			if ((doclkh < oldlkh) && (conv > 1e-6)){
				printf("ooops 2, %lf %lf, %lf %d\n", doclkh, oldlkh, conv, iter);
			}
			if ((iter > model->T) || (conv < CONVERGED))
				break;

			oldlkh = doclkh;
			iter += 1;
		}while(1);

	}

	if (test == 0){
		//update ss
		for (i = 0; i < doc->length; i++){
			n = doc->words[i];
			cnt = (double) doc->counts[i];

			for (j = 0; j < model->m; j++){
				ss->t[j][n] += cnt*doc->phi[i][j];
				ss->sumt[j] += cnt*doc->phi[i][j];
			}
		}
		for (j = 0; j < model->m; j++){
			ss->alpha += gsl_sf_psi(doc->gamma[j]);
		}
		ss->alpha -= model->m*gsl_sf_psi(doc->sumgamma);
	}

	return(doclkh);
}




void test(char* dataset, char* settings_file, char* model_name, char* dir)
{

	FILE* lhood_fptr;
	FILE* fp;
	char string[100];
	char filename[100];
	char lblfile[100];
	int iteration;
	int d, c, ntopics, nclasses, ndocs, i, j;
	double wrdlkh, temp;
	document* doc;
	sslda_corpus* corpus;
	sslda_model *model = NULL;
	sslda_ss *ss = NULL;
	time_t t1,t2;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &ndocs);
	fscanf(fp, "N %d\n", &d);
	fscanf(fp, "T %d\n", &MAXITER);
	fscanf(fp, "alpha %lf\n", &temp);
	fscanf(fp, "converged %lf\n", &CONVERGED);
	fclose(fp);

	corpus = read_data(dataset, nclasses, ndocs, 0, lblfile, ntopics);


	mkdir(dir, S_IRUSR|S_IWUSR|S_IXUSR);
	model = new_sslda_model(corpus, settings_file);
	ss = new_sslda_ss(model);

	corpus->nterms = model->n;
	corpus->ndocs = model->D;
	load_model(model, corpus, ss, model_name);
	//init
	double nd, cnt, h_i;
	//int n;
	for (d = 0; d < model->D; d++){
		doc = &(corpus->docs[d]);

		nd = (double) doc->total;
		for (j = 0; j < model->m; j++){
			doc->gamma[j] = model->alpha;
			doc->sumphi[j] = 0.0;
		}
		for (c = 0; c < model->c; c++){
			doc->xi[c] = 0.0;
		}
		for (i = 0; i < doc->length; i++){
			//n = doc->words[i];
			cnt = (double) doc->counts[i];

			for (j = 0; j < model->m; j++){
				doc->phi[i][j] = 1.0/model->m;
				doc->sumphi[j] += cnt*doc->phi[i][j];
				doc->gamma[j] += cnt*doc->phi[i][j];
			}

			for (c = 0; c < model->c; c++){
				h_i =  0.0;
				for (j = 0; j < model->m; j++){
					h_i += doc->phi[i][j]*exp(cnt*model->w[j][c]/nd);
				}
				doc->xi[c] += log(h_i);
			}
		}
		doc->sumgamma = 0.0;
		for (j = 0; j < model->m; j++){
			doc->sumgamma += doc->gamma[j];
		}
	}

	// set up the log likelihood log file
	sprintf(string, "%s/test-lhood.dat", dir);
	lhood_fptr = fopen(string, "w");

	iteration = 0;
	wrdlkh = 0.0;

	time(&t1);
	// int tt;

	double** b = malloc(sizeof(double*)*model->c);
	for (c = 0; c < model->c; c++){
		b[c] = malloc(sizeof(double)*corpus->ndocs);
	}

	sprintf(filename, "%s/testfinal", dir);
	write_sslda_model(corpus, model, ss, filename);

	double lkh = 0.0;
	double normsum, theta;
	wrdlkh = 0.0;
	for (d = 0; d < corpus->ndocs; d++){

		doc = &(corpus->docs[d]);
		lkh += doc_estep(doc, model, ss, d, 1);


		//wrdlkh
		for (i = 0; i < doc->length; i++){

			temp = 0.0;
			for (j = 0; j < model->m; j++){
				theta = doc->gamma[j]/doc->sumgamma;
				temp += theta*model->beta[j][doc->words[i]];
			}
			wrdlkh += ((double)doc->counts[i])*log(temp);
		}
		nd = (double) doc->total;
		for (c = 0; c < model->c; c++){
			temp = 0.0;
			for (j = 0; j < model->m; j++){
				temp += model->w[j][c]*doc->sumphi[j]/nd;
			}

			if (temp < 0){
				b[c][d] = exp(temp)/(1+exp(temp));
			}else{
				b[c][d] = 1.0/(1+exp(-temp));
			}

		}

	}


	time(&t2);

	fprintf(lhood_fptr, "%d %e %5ld\n",iteration, wrdlkh, (int)t2-t1);
	fflush(lhood_fptr);

	//mu
	sprintf(filename, "%s/testfinal.b", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (c = 0; c < model->c; c++){
			fprintf(fp, "%5.10lf ", b[c][d]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	//mu
	sprintf(filename, "%s/testfinal.gamma", dir);
	fp = fopen(filename, "w");
	for (d = 0; d < model->D; d++){
		for (j = 0; j < model->m; j++){
			fprintf(fp, "%5.10lf ", corpus->docs[d].gamma[j]);///corpus->docs[d].sumgamma);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

}



sslda_model* new_sslda_model(sslda_corpus* corpus, char* settings_file)
{
	int n, j, c, ntopics, nclasses;
	//double temp1, temp2;
	FILE* fp;

	printf("loading %s\n", settings_file);
	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fclose(fp);

	sslda_model* model = malloc(sizeof(sslda_model));
	model->c = nclasses;
	model->m = ntopics;
	model->D = 0;
	model->n = 0;
	model->T = 0;
	model->alpha = 0.0;

	fp = fopen(settings_file, "r");
	fscanf(fp, "M %d\n", &ntopics);
	fscanf(fp, "C %d\n", &nclasses);
	fscanf(fp, "D %d\n", &model->D);
	fscanf(fp, "N %d\n", &model->n);
	fscanf(fp, "T %d\n", &model->T);
	fscanf(fp, "alpha %lf\n", &model->alpha);
	fclose(fp);

	model->beta = malloc(sizeof(double*)*model->m);
	model->logbeta = malloc(sizeof(double*)*model->m);
	model->w = malloc(sizeof(double*)*model->m);
	for (j = 0; j < model->m; j++){
		model->beta[j] = malloc(sizeof(double)*model->n);
		model->logbeta[j] = malloc(sizeof(double)*model->n);
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = 0.0;
			model->logbeta[j][n] = 0.0;
		}
		model->w[j] = malloc(sizeof(double)*model->c);
		for (c = 0; c < model->c; c++){
			model->w[j][c] = 0.0;
		}
	}

	return(model);
}

sslda_ss * new_sslda_ss(sslda_model* model){

	int n, j, c;

	sslda_ss* ss = malloc(sizeof(sslda_ss));

	ss->t = malloc(sizeof(double*)*model->m);
	ss->sumt = malloc(sizeof(double)*model->m);
	ss->oldphi = malloc(sizeof(double)*model->m);

	for (j = 0; j < model->m; j++){
		ss->oldphi[j] = 0.0;
		ss->sumt[j] = 0.0;
		ss->t[j] = malloc(sizeof(double)*model->n);
		for(n = 0; n < model->n; n++){
			ss->t[j][n] = 0.0;
			ss->sumt[j] = 0.0;
		}

	}

	ss->oldxi = malloc(sizeof(double)*model->c);
	ss->xi = malloc(sizeof(double)*model->c);
	ss->hi = malloc(sizeof(double)*model->c);
	for (c = 0; c < model->c; c++){
		ss->oldxi[c] = 0.0;
		ss->xi[c] = 0.0;
		ss->hi[c] = 0.0;
	}

	return(ss);
}




sslda_corpus* read_data(const char* data_filename, int nclasses, int ndocs,
		int lblchck, const char* lbl_filename, int ntopics)
{
	FILE *fileptr;
	int Ls, word, nd, nw, lbl;
	int i, c, j, count;
	sslda_corpus* corpus;

	printf("reading data from %s\n", data_filename);
	corpus = malloc(sizeof(sslda_corpus));
	//corpus->docs = 0;
	corpus->docs = (document *) malloc(sizeof(document)*(ndocs));
	corpus->nterms = 0.0;
	corpus->ndocs = 0.0;
	corpus->w = malloc(sizeof(double)*ntopics);
	corpus->grad = malloc(sizeof(double)*ntopics);

	fileptr = fopen(data_filename, "r");
	nd = 0; nw = 0;
	while ((fscanf(fileptr, "%10d ", &Ls) != EOF)){ // number of sentences
		corpus->docs[nd].length = Ls;
		corpus->docs[nd].total = 0;

		corpus->docs[nd].words = malloc(sizeof(int)*Ls);
		corpus->docs[nd].counts = malloc(sizeof(int)*Ls);

		corpus->docs[nd].phi = malloc(sizeof(double*)*Ls);
		corpus->docs[nd].sumphi = malloc(sizeof(double)*ntopics);
		corpus->docs[nd].gamma = malloc(sizeof(double)*ntopics);
		corpus->docs[nd].sumgamma = 0.0;
		corpus->docs[nd].b = malloc(sizeof(int)*nclasses);
		corpus->docs[nd].xi = malloc(sizeof(double)*nclasses);
		for (c = 0; c < nclasses; c++){
			corpus->docs[nd].b[c] = -1;
			corpus->docs[nd].xi[c] = 0.0;
		}
		for (j = 0; j < ntopics; j++){
			corpus->docs[nd].sumphi[j] = 0.0;
			corpus->docs[nd].gamma[j] = 0.0;
		}
		for (i = 0; i < Ls; i++){
			fscanf(fileptr, "%10d:%10d ", &word, &count);
			corpus->docs[nd].words[i] = word;
			//count = 1;
			corpus->docs[nd].counts[i] = count;
			corpus->docs[nd].total += count;
			corpus->docs[nd].phi[i] = malloc(sizeof(double)*ntopics);
			for (j = 0; j < ntopics; j++)
				corpus->docs[nd].phi[i][j] = 0.0;

			if (word >= nw) { nw = word + 1; }
		}
		nd++;
	}
	fclose(fileptr);
	corpus->ndocs = nd;
	corpus->nterms = nw;
	printf("number of docs    : %d\n", nd);
	printf("number of terms   : %d\n", nw);

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


void write_sslda_model(sslda_corpus * corpus, sslda_model * model, sslda_ss* ss, char * root)
{
	char filename[200];
	FILE* fileptr;
	int n, j, c;
	//document* doc;


	//beta
	sprintf(filename, "%s.beta", root);
	fileptr = fopen(filename, "w");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fprintf(fileptr, "%.10lf ",model->logbeta[j][n]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//mu
	sprintf(filename, "%s.w", root);
	fileptr = fopen(filename, "w");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fprintf(fileptr, "%5.10lf ", model->w[j][c]);
		}
		fprintf(fileptr, "\n");
	}
	fclose(fileptr);

	//mu
	sprintf(filename, "%s.alpha", root);
	fileptr = fopen(filename, "w");
	fprintf(fileptr, "%lf ", model->alpha);
	fclose(fileptr);

}

void load_model(sslda_model* model, sslda_corpus* corpus, sslda_ss* ss, char* model_root){

	char filename[100];
	FILE* fileptr;
	int j, n, c;
	//float x;
	double y;

	sprintf(filename, "%s.beta", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (n = 0; n < model->n; n++){
		for (j = 0; j < model->m; j++){
			fscanf(fileptr, " %lf", &y);
			model->logbeta[j][n] = y;
			model->beta[j][n] = exp(y);
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.w", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	for (j = 0; j < model->m; j++){
		for (c = 0; c < model->c; c++){
			fscanf(fileptr, "%lf ", &y);
			model->w[j][c] = y;
		}
	}
	fclose(fileptr);

	sprintf(filename, "%s.alpha", model_root);
	printf("loading %s\n", filename);
	fileptr = fopen(filename, "r");
	fscanf(fileptr, "%lf ", &model->alpha);
	model->logalpha = log(model->alpha);
	fclose(fileptr);


	//return(model);
}



void random_initialize_model(sslda_corpus* corpus, sslda_model * model, sslda_ss* ss){

	int i, d, n, j, c;
	double normsum, cnt, nd, h_i;
	document* doc;


	//init sample beta, mu
	for (j = 0; j < model->m; j++){
		normsum = 0.0;
		for (n = 0; n < model->n; n++){
			model->beta[j][n] = gsl_ran_flat (r, 0, 1);
			normsum += model->beta[j][n];
		}
		for (n = 0; n < model->n; n++){
			model->beta[j][n] /= normsum;
			if (model->beta[j][n] == 0) model->beta[j][n] = EPS;
			model->logbeta[j][n] = log(model->beta[j][n]);
		}
		for (c = 0; c < model->c; c++){
			model->w[j][c] = gsl_ran_flat (r, 0, 1);
		}
	}
	for (d = 0; d < model->D; d++){
		doc = &(corpus->docs[d]);

		nd = (double) doc->total;
		for (j = 0; j < model->m; j++){
			doc->gamma[j] = model->alpha;
			doc->sumphi[j] = 0.0;
		}
		for (c = 0; c < model->c; c++){
			doc->xi[c] = 0.0;
		}
		for (i = 0; i < doc->length; i++){
			n = doc->words[i];
			cnt = (double) doc->counts[i];

			for (j = 0; j < model->m; j++){
				doc->phi[i][j] = 1.0/model->m;
				doc->sumphi[j] += cnt*doc->phi[i][j];
				doc->gamma[j] += cnt*doc->phi[i][j];
			}

			for (c = 0; c < model->c; c++){
				h_i =  0.0;
				for (j = 0; j < model->m; j++){
					h_i += doc->phi[i][j]*exp(cnt*model->w[j][c]/nd);
				}
				doc->xi[c] += log(h_i);
			}
		}
		doc->sumgamma = 0.0;
		for (j = 0; j < model->m; j++){
			doc->sumgamma += doc->gamma[j];
		}

	}

}



