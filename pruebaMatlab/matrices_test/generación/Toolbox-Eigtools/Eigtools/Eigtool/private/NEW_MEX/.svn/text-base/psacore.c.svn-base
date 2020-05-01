/* Version 2.0r1, 24th June 2002
   Copyright 2002 by Tom Wright */

#include "mex.h"

typedef double doublereal;
typedef struct { doublereal r, i; } doublecomplex;
typedef int integer;

/* Table of constant values */

static doublecomplex c_b1 = {1.,0.};
static integer c__1 = 1;

/* Integer min function */
int imin(a,b) 
     int a,b;
{
  if (a<b) return a;
  else return b;
}

/* Integer max function */
int imax(a,b)
     int a,b;
{
  if (a>b) return a;
  else return b;
}

/* Subroutine for calculating Given's rotation complex */
/*     Taken from p216 of Golub & van Loan */
void cgivens(a, b, c, s)
doublecomplex *a, *b, *c, *s;
{
    /* Builtin functions */
    double sqrt();

    /* Local variables */
    doublereal tor;

    if (b->r == 0. && b->i == 0.) {
	c->r = 1., c->i = 0.;
	s->r = 0., s->i = 0.;
    } else {
	tor = sqrt(a->r*a->r + a->i*a->i + b->r*b->r + b->i*b->i);
        s->r = -b->r/tor;
        s->i = -b->i/tor;

        c->r = a->r/tor;
        c->i = -a->i/tor;
    }
} /* cgivens */


/* Subroutine for calculating the R part of the QR factorisation */
/*    for an upper Hessenberg matrix (complex). */
/*    Taken from p228 of Golub & van Loan */

void chessqr(h, m, n)
doublecomplex *h;
integer *m, *n;
{
    /* Local variables */
    doublecomplex c;
    integer i, j;
    doublecomplex s, t1, t2;

    /* Function Body */
    for (j = 0; j < *m-1; j++) {

	cgivens(&h[j+j*(*m)], &h[j+1+j*(*m)], &c, &s);

	for (i = j; i < *n; i++) {

	    t1.r = h[j+i*(*m)].r, t1.i = h[j+i*(*m)].i;
	    t2.r = h[j+1+i*(*m)].r, t2.i = h[j+1+i*(*m)].i;

            h[j+i*(*m)].r = c.r*t1.r-c.i*t1.i - s.r*t2.r - s.i*t2.i;
            h[j+i*(*m)].i = c.i*t1.r+c.r*t1.i + s.i*t2.r - s.r*t2.i;

            h[j+1+i*(*m)].r = s.r*t1.r-s.i*t1.i + c.r*t2.r + c.i*t2.i;
            h[j+1+i*(*m)].i = s.i*t1.r+s.r*t1.i - c.i*t2.r + c.r*t2.i;

	}
    }
} /* chessqr */


void clanczos(A, m, n, q, smin, tol, v, dh, maxk)
  doublecomplex *A;
  integer m, n, maxk;
  doublecomplex *q, *v;
  doublereal *smin, *tol, *dh;
{

    double sqrt(), fabs();

    doublereal beta;
    integer info;
    integer i, k;
    doublereal alpha;
    doublecomplex *qold;
    doublereal *odh, *tdh, *todh;
    doublereal dummy, sig, sigold; 

    /* Extract the addresses of these locations from the pointer passed in */
    qold = &v[n];
    odh = &dh[maxk];
    tdh = &odh[maxk];
    todh = &tdh[maxk];

    for (i = 0; i < n; i++) {
	qold[i].r = 0.;
        qold[i].i = 0.;
    }

    beta = 0.;
    sigold = 0.;

    for (k = 1; k <= maxk; k++) {

/* q will be overwritten below, so need to save it */
        for (i = 0; i < n; i++) {
	    v[i].r = q[i].r;
            v[i].i = q[i].i;
        }

       	ztrsm("L", "U", "C", "N", &n, &c__1, &c_b1, A, &m, v, &n);
	ztrsm("L", "U", "N", "N", &n, &c__1, &c_b1, A, &m, v, &n);

	alpha = 0.;
        for (i = 0; i < n; i++) {
            v[i].r -= beta*qold[i].r;
            v[i].i -= beta*qold[i].i;
/* This is a plus here because we're conjugating v */
            alpha += v[i].r*q[i].r + v[i].i*q[i].i;
	}

        for (i = 0; i < n; i++) {
            v[i].r -= alpha*q[i].r;
            v[i].i -= alpha*q[i].i;
	}

	beta = 0.;
        for (i = 0; i < n; i++) {
            beta += v[i].r*v[i].r + v[i].i*v[i].i;
	}
	beta = sqrt(beta);

        for (i = 0; i < n; i++) {
            qold[i].r = q[i].r;
            qold[i].i= q[i].i;
            q[i].r = v[i].r/beta;
            q[i].i = v[i].i/beta;
	}

/* Append the values to the matrix (and save them in the */
/* temporary storage for later) */
	odh[k-1] = beta;
	todh[k-1] = beta;
	dh[k-1] = alpha;
	tdh[k-1] = alpha;

/* Get the eigenvalues of H */
	dsteqr("N", &k, dh, odh, &dummy, &c__1, &dummy, &info);
	if (info != 0) {
	    sig = 1e308;
            mexWarnMsgTxt("sigmin set to smallest value possible.");
	    break;
	}

/* The eigenvalues are returned in ascending order */
	sig = dh[k-1];

/* Copy the values of H back from temporary storage */
        for (i = 0; i < k; i++) {
            dh[i] = tdh[i];
            odh[i] = todh[i];
	}

	if ((fabs(sigold/sig - 1.) < *tol)||(beta==0)) break;

	sigold = sig;
    }

    *smin = 1/sqrt(sig);

} /* clanczos */

void cpsacore(m, n, bw, q_pr, q_pi, smin, x, nx, y, ny, tol, T_pr, T_pi, S_pr, S_pi)
  integer m, n, bw;
  doublereal *smin, *x;
  integer nx;
  doublereal *y;
  integer ny;
  doublereal *tol, *T_pr, *T_pi, *S_pr, *S_pi, *q_pr, *q_pi;
{

    int const maxk=100;

    int i, j, k, l, info, lwork;
    doublecomplex *temp_q, *tau, *work, *Tc, *v;
    doublecomplex temp_work;
    doublecomplex z;
    doublereal *dh;

    /* Allocate memory for v and qold (used in clanczos) */
    v = (doublecomplex*)mxCalloc(2*n,sizeof(doublecomplex));
    if (v==NULL) mexErrMsgTxt("Memory allocation for vector v failed.");
  
    /* Allocate memory for dh et al. (used in clanczos) */
    dh = (doublereal*)mxCalloc(4*maxk,sizeof(doublereal));
    if (dh==NULL) mexErrMsgTxt("Memory allocation for vector dh failed.");

    /* If we're going to use zbandqrf (not hessqr) */
    if (bw>2) {
      /* Allocate memory for tau */
      tau = (doublecomplex*)mxCalloc(n,sizeof(doublecomplex));
      if (tau==NULL) mexErrMsgTxt("Memory allocation for tau failed.");

      /* Get optimal workspace size */
      lwork = -1;
      zbandqrf(&m, &n, &bw,Tc, &m, tau, &temp_work, &lwork, &info );
      lwork = (int)temp_work.r;

      /* Allocate memory for work */
      work = (doublecomplex*)mxCalloc(lwork,sizeof(doublecomplex));
      if (work==NULL) mexErrMsgTxt("Memory allocation for work failed.");
    }

    /* Allocate memory for the matrix (we need it in our format) */
    Tc = (doublecomplex*)mxCalloc(m*n,sizeof(doublecomplex));
    if (Tc==NULL) mexErrMsgTxt("Memory allocation for matrix failed.");

    /* Allocate memory for the vector */
    temp_q = (doublecomplex*)mxCalloc(n,sizeof(doublecomplex));
    if (temp_q==NULL) mexErrMsgTxt("Memory allocation for temp vec failed.");

    /* If the matrix is square, we do things differently; copy once here,
       then only update the diagonal below */
    if ((m==n) && (T_pi==NULL)){
      for (i = 0; i < n; i++) {
        for (l = 0; l <= i; l++) {
	  Tc[l+m*i].r = T_pr[l+m*i];
          Tc[l+m*i].i = 0.;
        }
      }
    }
    else if (m==n) {
      for (i = 0; i < n; i++) {
        for (l = 0; l <= i; l++) {
	  Tc[l+m*i].r = T_pr[l+m*i];
          Tc[l+m*i].i = T_pi[l+m*i];
        }
      }
    }

    /* Now loop over all the gridpoints provided */
    for (j = 0; j<ny; j++) {
      for (k = 0; k<nx; k++) {

	/* Set the complex shift */
        z.r = x[k];
        z.i = y[j];

	/* If m==n, only need to update the diagonal, since the matrix
	   is not overwritten in the routines */
        if ((m==n) && (T_pi==NULL)) {
	  for (i = 0; i < n; i++) {
	    /* Update the diagonal */
	    Tc[(m+1)*i].r = T_pr[(m+1)*i] - z.r;
	    Tc[(m+1)*i].i = -z.i;
          }
	}
	else if (m==n) { /* There is an imaginary bit */
	  for (i = 0; i < n; i++) {
	    /* Update the diagonal */
	    Tc[(m+1)*i].r = T_pr[(m+1)*i] - z.r;
	    Tc[(m+1)*i].i = T_pi[(m+1)*i] - z.i;
          }
	}
	else { /* Matrix is not square */
	  /* If there is no matrix S (i.e., use the identity) */
	  if ((S_pr==NULL) && (S_pi==NULL)) {
	    if (T_pi == NULL) {
	      for (i = 0; i < n; i++) {
		/* Update the entries above the diagonal */
		for (l = 0; l < i; l++) {
		  Tc[l+m*i].r = T_pr[l+m*i];
		  Tc[l+m*i].i = 0.;
		}
		/* Update the diagonal */
		Tc[(m+1)*i].r = T_pr[(m+1)*i] - z.r;
		Tc[(m+1)*i].i = -z.i;
		/* Update the entries below the diagonal */
		for (l = i+1; l < imin(i+bw,m); l++) {
		  Tc[l+m*i].r = T_pr[l+m*i];
		  Tc[l+m*i].i = 0.;
		}
	      }
	    }
	    else {  /* If T has an imaginary part */
	      for (i = 0; i < n; i++) {
		/* Update the entries above the diagonal */
		for (l = 0; l < i; l++) {
		  Tc[l+m*i].r = T_pr[l+m*i];
		  Tc[l+m*i].i = T_pi[l+m*i];
		}
		/* Update the diagonal */
		Tc[(m+1)*i].r = T_pr[(m+1)*i] - z.r;
		Tc[(m+1)*i].i = T_pi[(m+1)*i] - z.i;
		/* Update the entries below the diagonal */
		for (l = i+1; l < imin(i+bw,m); l++) {
		  Tc[l+m*i].r = T_pr[l+m*i];
		  Tc[l+m*i].i = T_pi[l+m*i];
		}
	      }
	    }
	  }
	  else {  /* If there is a matrix S */
	    if ((T_pi == NULL) && (S_pi == NULL)) {
	      for (i = 0; i < n; i++) {
		for (l = 0; l < imin(i+bw,m); l++) {
		  Tc[l+m*i].r = T_pr[l+m*i] - z.r*S_pr[l+m*i];
		  Tc[l+m*i].i = -z.i*S_pr[l+m*i];
		}
	      }
	    }
	    else if ((T_pi != NULL) && (S_pi == NULL)) {
	      for (i = 0; i < n; i++) {
		for (l = 0; l < imin(i+bw,m); l++) {
		  Tc[l+m*i].r = T_pr[l+m*i] - z.r*S_pr[l+m*i];
		  Tc[l+m*i].i = T_pi[l+m*i] - z.i*S_pr[l+m*i];
		}
	      }
	    }
	    else if ((T_pi == NULL) && (S_pi != NULL)) {
	      for (i = 0; i < n; i++) {
		for (l = 0; l < imin(i+bw,m); l++) {
		  Tc[l+m*i].r = T_pr[l+m*i] - z.r*S_pr[l+m*i] + z.i*S_pi[l+m*i];
		  Tc[l+m*i].i = -z.i*S_pr[l+m*i] - z.r*S_pi[l+m*i];
		}
	      }
	    }
	    else {
	      for (i = 0; i < n; i++) {
		for (l = 0; l < imin(i+bw,m); l++) {
		  Tc[l+m*i].r = T_pr[l+m*i] - z.r*S_pr[l+m*i] + z.i*S_pi[l+m*i];
		  Tc[l+m*i].i = T_pi[l+m*i] - z.i*S_pr[l+m*i] - z.r*S_pi[l+m*i];
		}
	      }
	    }
	  }
	}  /* Is matrix square */

	/* Restore the starting vector */
	if (q_pi!=NULL) {
	  for (i = 0; i < n; i++) {
	    temp_q[i].r = q_pr[i];
	    temp_q[i].i = q_pi[i];
	  }
	}
	else {
	  for (i = 0; i < n; i++) {
	    temp_q[i].r = q_pr[i];
	    temp_q[i].i = 0.;
	  }
	}

	/* Use hessqr if bandwidth is 2 (Hessenberg)---it's faster for some reason, 
	   as yet undetermined. Don't do this step at all if matrix is square (bw = 1) */
	if (bw==2) chessqr(Tc, &m, &n);
	else if (bw>2) zbandqrf(&m,&n,&bw,Tc, &m, tau, work, &lwork, &info );

	/* Now do the Lanczos iteration on the nxn triangle */
        clanczos(Tc, m, n, temp_q, &smin[j + k * ny],tol,v,dh,maxk);

      }
    }

    mxFree(v);
    mxFree(dh);
    mxFree(Tc);
    mxFree(temp_q);
    if (bw>2) { mxFree(tau); mxFree(work); }

} /* cpsacore */

/* Subroutine to handle the matlab interface */
void mexFunction ( int nlhs, mxArray *plhs[],
                   int nrhs, const mxArray *prhs[])
{
    double *T_pr, *T_pi, *S_pr, *S_pi, *q_pr, *q_pi, *smin_pr;
    int size, m, n, ibw, mq, nq, mx, nx, my, ny, mt, nt, ms, ns;
    int eye=0;
    doublereal *x, *y, *tol, *bw; 
    
    extern mxArray *mxCreateDoubleMatrix();
    extern void mexErrMsgTxt();
   /* extern integer mxGetM(), mxGetN(); */  /* commented out by mpe */
    extern double *mxGetPi(), *mxGetPr();
    extern bool mxIsNumeric(), mxIsComplex();
    extern void *mxCalloc();

    if (nrhs != 7) {
        mexErrMsgTxt("Seven inputs required - Band mtx1,Band mtx2,vec,x,y,tol,bandwidth");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Only one output required.");
    }

    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    size = m * n;

    if (m<n) {
        mexErrMsgTxt("m must be greater than or equal to n");
    }

    if ((m==1) && (n==1)) {
              mexErrMsgTxt("Sorry, this routine doesn't work with 1x1 matrices.");
    }

    ms = mxGetM(prhs[1]);
    ns = mxGetN(prhs[1]);
    /* If input is 1x1, use Identity matrix for S */
    if ((ms==1) && (ns==1)) {
      eye = 1;
    }
    else if ((ms!=m) || (ns!=n)) {
        mexErrMsgTxt("Dimensions of matrices S and T do not match.");
    }

    if ((m==n) && ((ms!=1) || (ns!=1))) {
        mexErrMsgTxt("Sorry, routine doesn't work for square, generalised problems yet.");
    }

    mq = mxGetM(prhs[2]);
    nq = mxGetN(prhs[2]);
    if ((mq!=n) || (nq!=1)) {
        mexErrMsgTxt("Dimensions of matrix and vector do not match.");
    }

    mx = mxGetM(prhs[3]);
    nx = mxGetN(prhs[3]);
    if (mx!=1) {
        mexErrMsgTxt("x must be no. x grid points by 1.");
    }

    my = mxGetM(prhs[4]);
    ny = mxGetN(prhs[4]);
    if (my!=1) {
        mexErrMsgTxt("y must be no. y grid points by 1.");
    }

    mt = mxGetM(prhs[5]);
    nt = mxGetN(prhs[5]);
    if ((mt!=1)||(nt!=1)) {
        mexErrMsgTxt("tol must be 1 by 1.");
    }

    mt = mxGetM(prhs[6]);
    nt = mxGetN(prhs[6]);
    if ((mt!=1)||(nt!=1)) {
        mexErrMsgTxt("bandwidth must be 1 by 1.");
    }

    if ((mxIsNumeric(prhs[0]) == 0)||(mxIsNumeric(prhs[1]) == 0)||(mxIsNumeric(prhs[2]) == 0)||(mxIsNumeric(prhs[3]) == 0)||(mxIsNumeric(prhs[4]) == 0)||(mxIsNumeric(prhs[5]) == 0)||(mxIsNumeric(prhs[6]) == 0)) {
        mexErrMsgTxt("Inputs must be numeric arrays.");
    }

    /* Don't acutally copy here - it's done in cpsacore */
    if (mxIsComplex(prhs[0]) != 1) {
        T_pr = mxGetPr(prhs[0]);
        T_pi = NULL;
    } else {
        T_pr = mxGetPr(prhs[0]);
        T_pi = mxGetPi(prhs[0]);
    }

    if (eye!=1) {
    /* Don't acutally copy here - it's done in cpsacore */
      if (mxIsComplex(prhs[1]) != 1) {
          S_pr = mxGetPr(prhs[1]);
          S_pi = NULL;
      } else {
          S_pr = mxGetPr(prhs[1]);
          S_pi = mxGetPi(prhs[1]);
      }
    }
    else { S_pr = NULL; S_pi = NULL; }

    /* If the vector data is real... */
    if (mxIsComplex(prhs[2]) != 1) {
        q_pr = mxGetPr(prhs[2]);
        q_pi = NULL;
    } else {
        q_pr = mxGetPr(prhs[2]);
        q_pi = mxGetPi(prhs[2]);
    }

    x = mxGetPr(prhs[3]);
    y = mxGetPr(prhs[4]);
    tol = mxGetPr(prhs[5]);
    bw = mxGetPr(prhs[6]);
    ibw = (int)(*bw);
    if ((ibw>m)||(ibw<1)) mexErrMsgTxt("Must have 1 <= bandwidth <= m (number of rows)");

    plhs[0] = mxCreateDoubleMatrix(ny, nx, mxREAL);
    smin_pr = mxGetPr(plhs[0]);
    
    cpsacore(m, n, ibw, q_pr, q_pi, smin_pr, x, nx, y, ny, tol, T_pr, T_pi, S_pr, S_pi);

}







