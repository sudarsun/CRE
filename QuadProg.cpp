#include "QuadProg.hpp"
#include <cstdio>
#include <memory>

/*
   Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.

   File:      qo1.c

   Purpose: To demonstrate how to solve a quadratic optimization
              problem using the MOSEK API.
 */

#include <stdio.h>

#include "mosek.h" /* Include the MOSEK definition file. */

#define NUMCON 1   /* Number of constraints.             */
//#define NUMVAR 3   /* Number of variables.               */
#define NUMANZ 3   /* Number of non-zeros in A.           */
//#define NUMQNZ 4   /* Number of non-zeros in Q.           */

static void MSKAPI printstr(void *handle,
                            MSKCONST char str[])
{
	//fprintf( stderr, "%s", str );

} /* printstr */

int QuadProg( const Qdata &q, double *c, int classes, real_array &outValues )
{
	int NUMVAR = classes;
//  double        c[]   = {0.0,-1.0,0.0};

  MSKboundkeye  bkc[] = {MSK_BK_LO};
  double        blc[] = {1.0};
  double        buc[] = {+MSK_INFINITY};

	MSKboundkeye *bkx = new MSKboundkeye[classes];
	double *blx = new double[classes],
	       *bux = new double[classes];

	MSKint32t 	*aptrb = new MSKint32t[classes],
				*aptre = new MSKint32t[classes],
				*asub = new MSKint32t[classes];
	double      *aval = new double[classes];

	// automatic memory releasers
	std::auto_ptr<MSKint32t> j1(aptrb), j2(aptre), j3(asub);
	std::auto_ptr<double> j4(aval), j5(blx), j6(bux);
	std::auto_ptr<MSKboundkeye> j7(bkx);

	for ( int i = 0; i < classes; ++i )
	{
		bkx[i] = MSK_BK_LO;
		blx[i] = 0.0;
		bux[i] = +MSK_INFINITY;

		aptrb[i] = i;
		aptre[i] = i+1;
		asub[i] = 0;
		aval[i] = 1.0;
	}

	MSKint32t *qsubi = q.qsubi;
	MSKint32t *qsubj = q.qsubj;
	double    *qval = q.qval;
	int NUMQNZ = q.nonzeros;

//	for ( int p = 0; p < NUMQNZ; ++p)
//		std::cout << "(" << qsubi[p] << "," << qsubj[p] << ") = " << qval[p] << std::endl;

  MSKint32t     i,j;
  double        xx[NUMVAR];

  MSKenv_t      env = NULL;
  MSKtask_t     task = NULL;
  MSKrescodee   r;

  /* Create the mosek environment. */
  r = MSK_makeenv(&env,NULL);

  if ( r==MSK_RES_OK )
  {
    /* Create the optimization task. */
    r = MSK_maketask(env,NUMCON,NUMVAR,&task);

    if ( r==MSK_RES_OK )
    {
      r = MSK_linkfunctotaskstream(task,MSK_STREAM_LOG,NULL,printstr);

      /* Append 'NUMCON' empty constraints.
       The constraints will initially have no bounds. */
      if ( r == MSK_RES_OK )
        r = MSK_appendcons(task,NUMCON);

      /* Append 'NUMVAR' variables.
       The variables will initially be fixed at zero (x=0). */
      if ( r == MSK_RES_OK )
        r = MSK_appendvars(task,NUMVAR);

      /* Optionally add a constant term to the objective. */
      if ( r ==MSK_RES_OK )
        r = MSK_putcfix(task,0.0);
      for(j=0; j<NUMVAR && r == MSK_RES_OK; ++j)
      {
        /* Set the linear term c_j in the objective.*/
        if(r == MSK_RES_OK)
          r = MSK_putcj(task,j,c[j]);

        /* Set the bounds on variable j.
         blx[j] <= x_j <= bux[j] */
        if(r == MSK_RES_OK)
          r = MSK_putvarbound(task,
                              j,           /* Index of variable.*/
                              bkx[j],      /* Bound key.*/
                              blx[j],      /* Numerical value of lower bound.*/
                              bux[j]);     /* Numerical value of upper bound.*/

        /* Input column j of A */
        if(r == MSK_RES_OK)
          r = MSK_putacol(task,
                          j,                 /* Variable (column) index.*/
                          aptre[j]-aptrb[j], /* Number of non-zeros in column j.*/
                          asub+aptrb[j],     /* Pointer to row indexes of column j.*/
                          aval+aptrb[j]);    /* Pointer to Values of column j.*/

      }

      /* Set the bounds on constraints.
         for i=1, ...,NUMCON : blc[i] <= constraint i <= buc[i] */
      for(i=0; i<NUMCON && r==MSK_RES_OK; ++i)
        r = MSK_putconbound(task,
                            i,           /* Index of constraint.*/
                            bkc[i],      /* Bound key.*/
                            blc[i],      /* Numerical value of lower bound.*/
                            buc[i]);     /* Numerical value of upper bound.*/

      if ( r==MSK_RES_OK )
      {
        /*
         * The lower triangular part of the Q
         * matrix in the objective is specified.
         */

        /* Input the Q for the objective. */
        r = MSK_putqobj(task,NUMQNZ,qsubi,qsubj,qval);
      }

      if ( r==MSK_RES_OK )
      {
        MSKrescodee trmcode;

        /* Run optimizer */
        r = MSK_optimizetrm(task,&trmcode);

        /* Print a summary containing information
           about the solution for debugging purposes*/
        MSK_solutionsummary (task,MSK_STREAM_MSG);

        if ( r==MSK_RES_OK )
        {
          MSKsolstae solsta;
          int j;

          MSK_getsolsta (task,MSK_SOL_ITR,&solsta);

          switch(solsta)
          {
            case MSK_SOL_STA_OPTIMAL:
            case MSK_SOL_STA_NEAR_OPTIMAL:
              MSK_getxx(task,
                       MSK_SOL_ITR,    /* Request the interior solution. */
                       xx);

			  outValues.resize( NUMVAR );
              fprintf(stderr, "Optimal primal solution\n");
              for(j=0; j<NUMVAR; ++j)
			  {
                fprintf(stderr, "x[%d]: %e\n",j,xx[j]);
				outValues[j] = xx[j];
			  }

              break;
            case MSK_SOL_STA_DUAL_INFEAS_CER:
            case MSK_SOL_STA_PRIM_INFEAS_CER:
            case MSK_SOL_STA_NEAR_DUAL_INFEAS_CER:
            case MSK_SOL_STA_NEAR_PRIM_INFEAS_CER:
              fprintf(stderr,"Primal or dual infeasibility certificate found.\n");
              break;

            case MSK_SOL_STA_UNKNOWN:
              fprintf(stderr,"The status of the solution could not be determined.\n");
              break;
            default:
              fprintf(stderr,"Other solution status.");
              break;
          }
        }
        else
        {
          fprintf(stderr,"Error while optimizing.\n");
        }
      }

      if (r != MSK_RES_OK)
      {
        /* In case of an error print error code and description. */
        char symname[MSK_MAX_STR_LEN];
        char desc[MSK_MAX_STR_LEN];

        fprintf(stderr,"An error occurred while optimizing.\n");
        MSK_getcodedesc (r,
                         symname,
                         desc);
        fprintf(stderr,"Error %s - '%s'\n",symname,desc);
      }
    }
    MSK_deletetask(&task);
  }
  MSK_deleteenv(&env);

  return (r);
} /* main */
