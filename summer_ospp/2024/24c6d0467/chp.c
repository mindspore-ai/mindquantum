#include <stdio.h>

#define         CNOT		0
#define         HADAMARD	1
#define         PHASE		2
#define         MEASURE		3

using namespace std;

struct QProg
// Quantum circuit

{

	long n;         // # of qubits
	long T;         // # of gates
	char *a; // Instruction opcode
	long *b; // Qubit 1
	long *c; // Qubit 2 (target for CNOT)
	int DISPQSTATE; // whether to print the state (q for final state only, Q for every iteration)
	int DISPTIME; // whether to print the execution time
	int SILENT;         // whether NOT to print measurement results
	int	DISPPROG; // whether to print instructions being executed as they're executed
	int SUPPRESSM; // whether to suppress actual computation of determinate measurement results

};


struct QState
// Quantum state
{

	// To save memory and increase speed, the bits are packed 32 to an unsigned long
	long n;         // # of qubits
	unsigned long **x; // (2n+1)*n matrix for stabilizer/destabilizer x bits (there's one "scratch row" at
	unsigned long **z; // (2n+1)*n matrix for z bits                                                 the bottom)
	int *r;         // Phase bits: 0 for +1, 1 for i, 2 for -1, 3 for -i.  Normally either 0 or 2.
	unsigned long pw[32]; // pw[i] = 2^i
	long over32; // floor(n/8)+1

};

void error(int k)

{

	if (k==0) printf("\nSyntax: chp [-options] <filename> [input]\n");
	if (k==1) printf("\nFile not found\n");
	exit(0);

}



void cnot(struct QState *q, long b, long c)
// Apply a CNOT gate with control b and target c
{

	long i;
	long b5;
	long c5;
	unsigned long pwb;
	unsigned long pwc;

	b5 = b>>5;
	c5 = c>>5;
	pwb = q->pw[b&31];
	pwc = q->pw[c&31];
	for (i = 0; i < 2*q->n; i++)
	{
         if (q->x[i][b5]&pwb) q->x[i][c5] ^= pwc;
         if (q->z[i][c5]&pwc) q->z[i][b5] ^= pwb;
		 if ((q->x[i][b5]&pwb) && (q->z[i][c5]&pwc) &&
			 (q->x[i][c5]&pwc) && (q->z[i][b5]&pwb))
				q->r[i] = (q->r[i]+2)%4;
		if ((q->x[i][b5]&pwb) && (q->z[i][c5]&pwc) &&
			!(q->x[i][c5]&pwc) && !(q->z[i][b5]&pwb))
				q->r[i] = (q->r[i]+2)%4;
	}

	return;

}



void hadamard(struct QState *q, long b)

// Apply a Hadamard gate to qubit b

{

	long i;
	unsigned long tmp;
	long b5;
	unsigned long pw;

	b5 = b>>5;
	pw = q->pw[b&31];
	for (i = 0; i < 2*q->n; i++)
	{
         tmp = q->x[i][b5];
         q->x[i][b5] ^= (q->x[i][b5] ^ q->z[i][b5]) & pw;
         q->z[i][b5] ^= (q->z[i][b5] ^ tmp) & pw;
         if ((q->x[i][b5]&pw) && (q->z[i][b5]&pw)) q->r[i] = (q->r[i]+2)%4;
	}

	return;

}



void phase(struct QState *q, long b)

// Apply a phase gate (|0>->|0>, |1>->i|1>) to qubit b

{

	long i;
	long b5;
	unsigned long pw;

	b5 = b>>5;
	pw = q->pw[b&31];
	for (i = 0; i < 2*q->n; i++)
	{
         if ((q->x[i][b5]&pw) && (q->z[i][b5]&pw)) q->r[i] = (q->r[i]+2)%4;
         q->z[i][b5] ^= q->x[i][b5]&pw;
	}

	return;

}



void rowcopy(struct QState *q, long i, long k)

// Sets row i equal to row k

{

	long j;

	for (j = 0; j < q->over32; j++)
	{
         q->x[i][j] = q->x[k][j];
         q->z[i][j] = q->z[k][j];
	}
	q->r[i] = q->r[k];

	return;

}



void rowswap(struct QState *q, long i, long k)

// Swaps row i and row k

{

	rowcopy(q, 2*q->n, k);
	rowcopy(q, k, i);
	rowcopy(q, i, 2*q->n);

	return;

}



void rowset(struct QState *q, long i, long b)

// Sets row i equal to the bth observable (X_1,...X_n,Z_1,...,Z_n)

{

	long j;
	long b5;
	unsigned long b31;

	for (j = 0; j < q->over32; j++)
	{
         q->x[i][j] = 0;
         q->z[i][j] = 0;
	}
	q->r[i] = 0;
	if (b < q->n)
	{
         b5 = b>>5;
         b31 = b&31;
         q->x[i][b5] = q->pw[b31];
	}
	else
	{
         b5 = (b - q->n)>>5;
         b31 = (b - q->n)&31;
         q->z[i][b5] = q->pw[b31];
	}

	return;

}



int clifford(struct QState *q, long i, long k)

// Return the phase (0,1,2,3) when row i is LEFT-multiplied by row k

{

	long j;
	long l;
	unsigned long pw;
	long e=0; // Power to which i is raised

	for (j = 0; j < q->over32; j++)
         for (l = 0; l < 32; l++)
         {
                 pw = q->pw[l];
                 if ((q->x[k][j]&pw) && (!(q->z[k][j]&pw))) // X
                 {
                         if ((q->x[i][j]&pw) && (q->z[i][j]&pw)) e++;         // XY=iZ
                         if ((!(q->x[i][j]&pw)) && (q->z[i][j]&pw)) e--;         // XZ=-iY
                 }
                 if ((q->x[k][j]&pw) && (q->z[k][j]&pw))                                 // Y
                 {
                         if ((!(q->x[i][j]&pw)) && (q->z[i][j]&pw)) e++;         // YZ=iX
                         if ((q->x[i][j]&pw) && (!(q->z[i][j]&pw))) e--;         // YX=-iZ
                 }
                 if ((!(q->x[k][j]&pw)) && (q->z[k][j]&pw))                         // Z
                 {
                         if ((q->x[i][j]&pw) && (!(q->z[i][j]&pw))) e++;         // ZX=iY
                         if ((q->x[i][j]&pw) && (q->z[i][j]&pw)) e--;         // ZY=-iX
                 }
         }

	e = (e+q->r[i]+q->r[k])%4;
	if (e>=0) return e;
         else return e+4;

}



void rowmult(struct QState *q, long i, long k)

// Left-multiply row i by row k

{

	long j;

	q->r[i] = clifford(q,i,k);
	for (j = 0; j < q->over32; j++)
	{
         q->x[i][j] ^= q->x[k][j];
         q->z[i][j] ^= q->z[k][j];
	}

	return;

}



void printstate(struct QState *q)

// Print the destabilizer and stabilizer for state q

{

	long i;
	long j;
	long j5;
	unsigned long pw;
	printf("n:%d\n", q->n);
	for (i = 0; i < 2*q->n; i++)
	{
         if (i == q->n)
         {
                 printf("\n");
                 for (j = 0; j < q->n+1; j++)
                         printf("-");
         }
         if (q->r[i]==2) printf("\n-");
         else printf("\n+");
         for (j = 0; j < q->n; j++)
         {
                 j5 = j>>5;
                 pw = q->pw[j&31];
                 if ((!(q->x[i][j5]&pw)) && (!(q->z[i][j5]&pw)))         printf("I");
                 if ((q->x[i][j5]&pw) && (!(q->z[i][j5]&pw)))         printf("X");
                 if ((q->x[i][j5]&pw) && (q->z[i][j5]&pw))                 printf("Y");
                 if ((!(q->x[i][j5]&pw)) && (q->z[i][j5]&pw))         printf("Z");
         }
	}
	printf("\n");

	return;

}



int measure(struct QState *q, long b, int sup)

// Measure qubit b
// Return 0 if outcome would always be 0
//                 1 if outcome would always be 1
//                 2 if outcome was random and 0 was chosen
//                 3 if outcome was random and 1 was chosen
// sup: 1 if determinate measurement results should be suppressed, 0 otherwise

{

	int ran = 0;
	long i;
	long p; // pivot row in stabilizer
	long m; // pivot row in destabilizer
	long b5;
	unsigned long pw;

	b5 = b>>5;
	pw = q->pw[b&31];
	for (p = 0; p < q->n; p++)         // loop over stabilizer generators
	{
         if (q->x[p+q->n][b5]&pw) ran = 1;         // if a Zbar does NOT commute with Z_b (the
         if (ran) break;                                                 // operator being measured), then outcome is random
	}

	// If outcome is indeterminate
	if (ran)
	{
         rowcopy(q, p, p + q->n);                         // Set Xbar_p := Zbar_p
         rowset(q, p + q->n, b + q->n);                 // Set Zbar_p := Z_b
         q->r[p + q->n] = 2*(rand()%2);                 // moment of quantum randomness
         for (i = 0; i < 2*q->n; i++)                 // Now update the Xbar's and Zbar's that don't commute with
                 if ((i!=p) && (q->x[i][b5]&pw))         // Z_b
                         rowmult(q, i, p);
         if (q->r[p + q->n]) return 3;
         else return 2;
	}

	// If outcome is determinate
	if ((!ran) && (!sup))
	{
         for (m = 0; m < q->n; m++)                         // Before we were checking if stabilizer generators commute
                 if (q->x[m][b5]&pw) break;                 // with Z_b; now we're checking destabilizer generators
         rowcopy(q, 2*q->n, m + q->n);
         for (i = m+1; i < q->n; i++)
                 if (q->x[i][b5]&pw)
                         rowmult(q, 2*q->n, i + q->n);
         if (q->r[2*q->n]) return 1;
         else return 0;
         /*for (i = m+1; i < q->n; i++)
                 if (q->x[i][b5]&pw)
                 {
                         rowmult(q, m + q->n, i + q->n);
                         rowmult(q, i, m);
                 }
         return (int)q->r[m + q->n];*/
	}

	return 0;

}



long gaussian(struct QState *q)

// Do Gaussian elimination to put the stabilizer generators in the following form:
// At the top, a minimal set of generators containing X's and Y's, in "quasi-upper-triangular" form.
// (Return value = number of such generators = log_2 of number of nonzero basis states)
// At the bottom, generators containing Z's only in quasi-upper-triangular form.

{

	long i = q->n;
	long k;
	long k2;
	long j;
	long j5;
	long g; // Return value
	unsigned long pw;

	for (j = 0; j < q->n; j++)
	{
         j5 = j>>5;
         pw = q->pw[j&31];
         for (k = i; k < 2*q->n; k++) // Find a generator containing X in jth column
                 if (q->x[k][j5]&pw) break;
         if (k < 2*q->n)
         {
                 rowswap(q, i, k);
                 rowswap(q, i-q->n, k-q->n);
                 for (k2 = i+1; k2 < 2*q->n; k2++)
                         if (q->x[k2][j5]&pw)
                         {
                                 rowmult(q, k2, i);         // Gaussian elimination step
                                 rowmult(q, i-q->n, k2-q->n);
                         }
                 i++;
         }
	}
	g = i - q->n;

	for (j = 0; j < q->n; j++)
	{
         j5 = j>>5;
         pw = q->pw[j&31];
         for (k = i; k < 2*q->n; k++) // Find a generator containing Z in jth column
                 if (q->z[k][j5]&pw) break;
         if (k < 2*q->n)
         {
                 rowswap(q, i, k);
                 rowswap(q, i-q->n, k-q->n);
                 for (k2 = i+1; k2 < 2*q->n; k2++)
                         if (q->z[k2][j5]&pw)
                         {
                                 rowmult(q, k2, i);
                                 rowmult(q, i-q->n, k2-q->n);
                         }
                 i++;
         }
	}

	return g;

}



long innerprod(struct QState *q1, struct QState *q2)

// Returns -1 if q1 and q2 are orthogonal
// Otherwise, returns a nonnegative integer s such that the inner product is (1/sqrt(2))^s

{

	return 0;

}



void printbasisstate(struct QState *q)

// Prints the result of applying the Pauli operator in the "scratch space" of q to |0...0>

{

	long j;
	long j5;
	unsigned long pw;
	int e = q->r[2*q->n];

	for (j = 0; j < q->n; j++)
	{
         j5 = j>>5;
         pw = q->pw[j&31];
         if ((q->x[2*q->n][j5]&pw) && (q->z[2*q->n][j5]&pw))         // Pauli operator is "Y"
                 e = (e+1)%4;
	}
	if (e==0) printf("\n +|");
	if (e==1) printf("\n+i|");
	if (e==2) printf("\n -|");
	if (e==3) printf("\n-i|");

	for (j = 0; j < q->n; j++)
	{
         j5 = j>>5;
         pw = q->pw[j&31];
         if (q->x[2*q->n][j5]&pw) printf("1");
                 else printf("0");
	}
	printf(">");

	return;

}



void seed(struct QState *q, long g)

// Finds a Pauli operator P such that the basis state P|0...0> occurs with nonzero amplitude in q, and
// writes P to the scratch space of q.  For this to work, Gaussian elimination must already have been
// performed on q.  g is the return value from gaussian(q).

{

	long i;
	long j;
	long j5;
	unsigned long pw;
	int f;
	long min;

	q->r[2*q->n] = 0;
	for (j = 0; j < q->over32; j++)
	{
         q->x[2*q->n][j] = 0;         // Wipe the scratch space clean
         q->z[2*q->n][j] = 0;
	}
	for (i = 2*q->n - 1; i >= q->n + g; i--)
	{
         f = q->r[i];
         for (j = q->n - 1; j >= 0; j--)
         {
                 j5 = j>>5;
                 pw = q->pw[j&31];
                 if (q->z[i][j5]&pw)
                 {
                         min = j;
                         if (q->x[2*q->n][j5]&pw) f = (f+2)%4;
                 }
         }
         if (f==2)
         {
                 j5 = min>>5;
                 pw = q->pw[min&31];
                 q->x[2*q->n][j5] ^= pw;         // Make the seed consistent with the ith equation
         }
	}

	return;

}



void printket(struct QState *q)

// Print the state in ket notation (warning: could be huge!)

{

	long g;         // log_2 of number of nonzero basis states
	unsigned long t;
	unsigned long t2;
	long i;

	g = gaussian(q);
	printf("\n2^%ld nonzero basis states", g);
	if (g > 31)
	{
         printf("\nState is WAY too big to print");
         return;
	}

	seed(q, g);
	printbasisstate(q);
	for (t = 0; t < q->pw[g]-1; t++)
	{
		t2 = t ^ (t+1);
         for (i = 0; i < g; i++)
                 if (t2 & q->pw[i])
                         rowmult(q, 2*q->n, q->n + i);
         printbasisstate(q);
	}
	printf("\n");

	return;

}



void runprog(struct QProg *h, struct QState *q)

// Simulate the quantum circuit

{

	long t;
	int m; // measurement result
	time_t tp;
	double dt;
	char mvirgin = 1;

	time(&tp);
	for (t = 0; t < h->T; t++)
	{
         if (h->a[t]==CNOT) cnot(q,h->b[t],h->c[t]);
         if (h->a[t]==HADAMARD) hadamard(q,h->b[t]);
         if (h->a[t]==PHASE) phase(q,h->b[t]);
         if (h->a[t]==MEASURE)
         {
                 if (mvirgin && h->DISPTIME)
                 {
                         dt = difftime(time(0),tp);
                         printf("\nGate time: %lf seconds", dt);
                         printf("\nTime per 10000 gates: %lf seconds", dt*10000.0f/(h->T - h->n));
                         time(&tp);
                 }
                 mvirgin = 0;
                 m = measure(q,h->b[t],h->SUPPRESSM);
                 if (!h->SILENT)
                 {
                         printf("\nOutcome of measuring qubit %ld: ", h->b[t]);
                         if (m>1) printf("%d (random)", m-2);
                         else printf("%d", m);
                 }
         }
         if (h->DISPPROG)
         {
                 if (h->a[t]==CNOT)         printf("\nCNOT %ld->%ld", h->b[t], h->c[t]);
                 if (h->a[t]==HADAMARD) printf("\nHadamard %ld", h->b[t]);
                 if (h->a[t]==PHASE)         printf("\nPhase %ld", h->b[t]);
         }
	}
	printf("\n");
	if (h->DISPTIME)
	{
         dt = difftime(time(0),tp);
         printf("\nMeasurement time: %lf seconds", dt);
         printf("\nTime per 10000 measurements: %lf seconds\n", dt*10000.0f/h->n);
	}
	if (h->DISPQSTATE)
	{
         printf("\nFinal state:");
         printstate(q);
         gaussian(q);
         printstate(q);
         printket(q);
	}
	return;

}



void preparestate(struct QState *q, char *s)
// 计算稳定子和非稳定子
// Prepare the initial state's "input"

{

	long l;
	long b;

	l = strlen(s);
	for (b = 0; b < l; b++)
	{
         if (s[b]=='Z')
         {
                 hadamard(q,b);
                 phase(q,b);
                 phase(q,b);
                 hadamard(q,b);
         }
         if (s[b]=='x') hadamard(q,b);
         if (s[b]=='X')
         {
                 hadamard(q,b);
                 phase(q,b);
                 phase(q,b);
         }
         if (s[b]=='y')
         {
                 hadamard(q,b);
                 phase(q,b);
         }
         if (s[b]=='Y')
         {
                 hadamard(q,b);
                 phase(q,b);
                 phase(q,b);
                 phase(q,b);                 
         }
	}

	return;

}



void initstae_(struct QState *q, long n, char *s)

// Initialize state q to have n qubits, and input specified by s
// 
{

	long i;
	long j;

	q->n = n;
	q->x = malloc((2*q->n + 1) * sizeof(unsigned long*));
	q->z = malloc((2*q->n + 1) * sizeof(unsigned long*));
	q->r = malloc((2*q->n + 1) * sizeof(int));
	q->over32 = (q->n>>5) + 1;
	q->pw[0] = 1;
	for (i = 1; i < 32; i++)
         q->pw[i] = 2*q->pw[i-1];
	for (i = 0; i < 2*q->n + 1; i++)
	{
         q->x[i] = malloc(q->over32 * sizeof(unsigned long));
         q->z[i] = malloc(q->over32 * sizeof(unsigned long));
         for (j = 0; j < q->over32; j++)
         {
                 q->x[i][j] = 0;
                 q->z[i][j] = 0;
         }
         if (i < q->n)
                 q->x[i][i>>5] = q->pw[i&31];
         else if (i < 2*q->n)
         {
                 j = i-q->n;
                 q->z[i][j>>5] = q->pw[j&31];
         }
         q->r[i] = 0;
	}
	if (s) preparestate(q, s);

	return;

}



void readprog(struct QProg *h, char *fn, char *params)
{

	long t;
	char fn2[255];
	FILE *fp;
	char c=0;
	long val;
	long l;

	h->DISPQSTATE = 0;
	h->DISPTIME = 0;
	h->SILENT = 0;
	h->DISPPROG = 0;
	h->SUPPRESSM = 0;
	if (params)
	{
         l = strlen(params);
         for (t = 1; t < l; t++)
         {
                 if ((params[t]=='q')||(params[t]=='Q')) h->DISPQSTATE = 1;
                 if ((params[t]=='p')||(params[t]=='P')) h->DISPPROG = 1;
                 if ((params[t]=='t')||(params[t]=='T')) h->DISPTIME = 1;
                 if ((params[t]=='s')||(params[t]=='S')) h->SILENT = 1;
                 if ((params[t]=='m')||(params[t]=='M')) h->SUPPRESSM = 1;
         }
	}
	sprintf(fn2, "%s", fn);
	fp = fopen(fn2, "r");
	if (!fp)
	{
        sprintf(fn2, "%s.chp", fn);
        fp = fopen(fn2, "r");
        if (!fp) error(1);
	}
	while (!feof(fp)&&(c!='#'))
        fscanf(fp, "%c", &c);
	if (c!='#') error(2);
	h->T = 0;
	h->n = 0;
	while (!feof(fp))
	{
        fscanf(fp, "%c", &c);
        if ((c=='\r')||(c=='\n'))
                 continue;
        fscanf(fp, "%ld", &val);
        if (val+1 > h->n) h->n = val+1;
        if ((c=='c')||(c=='C'))
        {
                fscanf(fp, "%ld", &val);
                if (val+1 > h->n) h->n = val+1;
        }
        h->T++;
	}
	fclose(fp);
	h->a = malloc(h->T * sizeof(char));
	h->b = malloc(h->T * sizeof(long));
	h->c = malloc(h->T * sizeof(long));
	fp = fopen(fn2, "r");
	while (!feof(fp)&&(c!='#'))
         fscanf(fp, "%c", &c);
	t=0;
	while (!feof(fp))
	{
         fscanf(fp, "%c", &c);
         if ((c=='\r')||(c=='\n'))
                 continue;
         if ((c=='c')||(c=='C')) h->a[t] = CNOT;
         if ((c=='h')||(c=='H')) h->a[t] = HADAMARD;
         if ((c=='p')||(c=='P')) h->a[t] = PHASE;
         if ((c=='m')||(c=='M')) h->a[t] = MEASURE;
         fscanf(fp, "%ld", &h->b[t]);
         if (h->a[t]==CNOT) fscanf(fp, "%ld", &h->c[t]);
         t++;
	}
	fclose(fp);
	return;
}



int main(int argc, char **argv)

{

	struct QProg *h;
	struct QState *q;
	int param=0; // whether there are command-line parameters
	srand(time(0));
	if (argc==1) error(0);
	if (argv[1][0]=='-') param = 1;
	h = malloc(sizeof(struct QProg));
	q = malloc(sizeof(struct QState));
	if (param) readprog(h,argv[2],argv[1]);
	else readprog(h,argv[1],NULL);
	if (argc==(3+param)) initstae_(q,h->n,argv[2+param]);
	else initstae_(q,h->n,NULL);
	runprog(h,q);

	return 0;

}