/* March 26, 2019

License: FreeBSD

To build and run:

	$ gcc -O0 -o eigenmath eigenmath.c -lm
	$ ./eigenmath

Press control-C to exit.

To run a script:

	$ ./eigenmath scriptfilename

A test script is available here:

	www.eigenmath.org/selftest
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <setjmp.h>
#include <unistd.h>
//#include <pthread.h>

#define TOS 1000000 // size of evaluation stack

#define NSYM 1000 // size of symbol table

// Symbolic expressions are built by connecting U structs.
//
// For example, (a * b + c) is built like this:
//
//           _______      _______                                _______
//          |CONS   |--->|CONS   |----------------------------->|CONS   |
//          |       |    |       |                              |       |
//          |_______|    |_______|                              |_______|
//              |            |                                      |
//           ___v___      ___v___      _______      _______      ___v___
//          |ADD    |    |CONS   |--->|CONS   |--->|CONS   |    |SYM c  |
//          |       |    |       |    |       |    |       |    |       |
//          |_______|    |_______|    |_______|    |_______|    |_______|
//                           |            |            |
//                        ___v___      ___v___      ___v___
//                       |MUL    |    |SYM a  |    |SYM b  |
//                       |       |    |       |    |       |
//                       |_______|    |_______|    |_______|

typedef struct U {
	union {
		struct {
			struct U *car;		// pointing down
			struct U *cdr;		// pointing right
		} cons;
		char *printname;
		char *str;
		struct tensor *tensor;
		struct {
			unsigned int *a, *b;	// rational number a over b
		} q;
		double d;
	} u;
	unsigned char k, tag;
} U;

// the following enum is for struct U, member k

enum {
	CONS,
	NUM,
	DOUBLE,
	STR,
	TENSOR,
	SYM,
};

// the following enum is for indexing the symbol table

enum {
	ABS,
	ADD,
	ADJ,
	AND,
	ARCCOS,
	ARCCOSH,
	ARCSIN,
	ARCSINH,
	ARCTAN,
	ARCTANH,
	ARG,
	ATOMIZE,
	BESSELJ,
	BESSELY,
	BINDING,
	BINOMIAL,
	CEILING,
	CHECK,
	CHOOSE,
	CIRCEXP,
	CLOCK,
	COEFF,
	COFACTOR,
	CONDENSE,
	CONJ,
	CONTRACT,
	COS,
	COSH,
	DECOMP,
	DEFINT,
	DEGREE,
	DENOMINATOR,
	DERIVATIVE,
	DET,
	DIM,
	DISPLAY,
	DIVISORS,
	DO,
	DOT,
	DRAW,
	EIGEN,
	EIGENVAL,
	EIGENVEC,
	ERF,
	ERFC,
	EVAL,
	EXP,
	EXPAND,
	EXPCOS,
	EXPSIN,
	FACTOR,
	FACTORIAL,
	FACTORPOLY,
	FILTER,
	FLOATF,
	FLOOR,
	FOR,
	GCD,
	HERMITE,
	HILBERT,
	IMAG,
	INDEX,
	INNER,
	INTEGRAL,
	INV,
	INVG,
	ISINTEGER,
	ISPRIME,
	LAGUERRE,
	LCM,
	LEADING,
	LEGENDRE,
	LOG,
	MAG,
	MOD,
	MULTIPLY,
	NOT,
	NROOTS,
	NUMBER,
	NUMERATOR,
	OPERATOR,
	OR,
	OUTER,
	POLAR,
	POWER,
	PRIME,
	PRINT,
	PRODUCT,
	QUOTE,
	QUOTIENT,
	RANK,
	RATIONALIZE,
	REAL,
	YYRECT,
	ROOTS,
	SETQ,
	SGN,
	SIMPLIFY,
	SIN,
	SINH,
	SQRT,
	STOP,
	SUBST,
	SUM,
	TAN,
	TANH,
	TAYLOR,
	TEST,
	TESTEQ,
	TESTGE,
	TESTGT,
	TESTLE,
	TESTLT,
	TRANSPOSE,
	UNIT,
	ZERO,

	MARK1,	// boundary (symbols above are functions)

	NATNUM,	// natural number
	NIL,
	PI,
	V0,
	V1,
	IU,

	MARK2,	// boundary (symbols above cannot be bound)

	METAA,
	METAB,
	METAX,
	SPECX,

	SYMBOL_A,
	SYMBOL_B,
	SYMBOL_C,
	SYMBOL_D,
	SYMBOL_I,
	SYMBOL_J,
	SYMBOL_N,
	SYMBOL_R,
	SYMBOL_S,
	SYMBOL_T,
	SYMBOL_X,
	SYMBOL_Y,
	SYMBOL_Z,

	AUTOEXPAND,
	BAKE,
	LAST,
	TRACE,
	TTY,

	MARK3,	// boundary (user defined symbols follow)
};

#define EXP1 NATNUM
#define MAXPRIMETAB 10000
#define MAXDIM 24

typedef struct tensor {
	int ndim;
	int dim[MAXDIM];
	int nelem;
	U *elem[1];
} T;

struct display {
	struct display *next;
	unsigned char type, attr;
	int h, w;
	int tot_h, tot_w;
	int len;
	unsigned char buf[0];
};

struct text_metric {
	float size;
	int ascent, descent, width, xheight, em;
};

#define zero binding[V0]
#define one binding[V1]
#define imaginaryunit binding[IU]

#define symbol(x) (symtab + (x))
#define push_symbol(x) push(symbol(x))
#define iscons(p) ((p)->k == CONS)
#define isrational(p) ((p)->k == NUM)
#define isdouble(p) ((p)->k == DOUBLE)
#define isnum(p) (isrational(p) || isdouble(p))
#define isstr(p) ((p)->k == STR)
#define istensor(p) ((p)->k == TENSOR)
#define issymbol(p) ((p)->k == SYM)
#define iskeyword(p) ((p)->k == SYM && (p) - symtab < MARK1)

#define car(p) (iscons(p) ? (p)->u.cons.car : symbol(NIL))
#define cdr(p) (iscons(p) ? (p)->u.cons.cdr : symbol(NIL))
#define caar(p) car(car(p))
#define cadr(p) car(cdr(p))
#define cdar(p) cdr(car(p))
#define cddr(p) cdr(cdr(p))
#define caadr(p) car(car(cdr(p)))
#define caddr(p) car(cdr(cdr(p)))
#define cadar(p) car(cdr(car(p)))
#define cdadr(p) cdr(car(cdr(p)))
#define cddar(p) cdr(cdr(car(p)))
#define cdddr(p) cdr(cdr(cdr(p)))
#define caaddr(p) car(car(cdr(cdr(p))))
#define cadadr(p) car(cdr(car(cdr(p))))
#define caddar(p) car(cdr(cdr(car(p))))
#define cdaddr(p) cdr(car(cdr(cdr(p))))
#define cadddr(p) car(cdr(cdr(cdr(p))))
#define cddddr(p) cdr(cdr(cdr(cdr(p))))
#define caddddr(p) car(cdr(cdr(cdr(cdr(p)))))
#define cadaddr(p) car(cdr(car(cdr(cdr(p)))))
#define cddaddr(p) cdr(cdr(car(cdr(cdr(p)))))
#define caddadr(p) car(cdr(cdr(car(cdr(p)))))
#define cdddaddr(p) cdr(cdr(cdr(car(cdr(cdr(p))))))
#define caddaddr(p) car(cdr(cdr(car(cdr(cdr(p))))))

#define isadd(p) (car(p) == symbol(ADD))
#define ispower(p) (car(p) == symbol(POWER))
#define isfactorial(p) (car(p) == symbol(FACTORIAL))

#define MSIGN(p) (((int *) (p))[-2])
#define MLENGTH(p) (((int *) (p))[-1])

#define MZERO(p) (MLENGTH(p) == 1 && (p)[0] == 0)
#define MEQUAL(p, n) (MLENGTH(p) == 1 && (long long) MSIGN(p) * (p)[0] == (n))

extern int tos;
extern int expanding;
extern int primetab[MAXPRIMETAB];
extern int esc_flag;
extern int draw_flag;
extern int trigmode;
extern int term_flag;

extern U symtab[];
extern U *binding[];
extern U *arglist[];
extern U *stack[];
extern U **frame;
extern U *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9;

extern jmp_buf draw_stop_return;

#define BLACK 0
#define BLUE 1
#define RED 2

void eval_abs(void);
void absval(void);
void absval_tensor(void);
void eval_add(void);
void add_terms(int n);
int cmp_terms(const void *q1, const void *q2);
int combine_terms(U **s, int n);
void push_terms(U *p);
void add(void);
void add_all(int k);
void subtract(void);
void eval_adj(void);
void adj(void);
U * alloc(void);
U * alloc_tensor(int nelem);
void gc(void);
void untag(U *p);
void alloc_mem(void);
void print_mem_info(void);
void append(void);
void eval_arccos(void);
void arccos(void);
void eval_arccosh(void);
void arccosh(void);
void eval_arcsin(void);
void arcsin(void);
void eval_arcsinh(void);
void arcsinh(void);
void eval_arctan(void);
void arctan(void);
void eval_arctanh(void);
void arctanh(void);
void eval_arg(void);
void arg(void);
void yyarg(void);
void eval_atomize(void);
void atomize(void);
void bake(void);
void polyform(void);
void bake_poly(void);
void bake_poly_term(int k);
void eval_besselj(void);
void besselj(void);
void yybesselj(void);
void eval_bessely(void);
void bessely(void);
void yybessely(void);
unsigned int * mnew(int n);
void mfree(unsigned int *p);
unsigned int * mint(int n);
unsigned int * mcopy(unsigned int *a);
int ge(unsigned int *a, unsigned int *b, int len);
void add_numbers(void);
void subtract_numbers(void);
void multiply_numbers(void);
void divide_numbers(void);
void invert_number(void);
int compare_rationals(U *a, U *b);
int compare_numbers(U *a, U *b);
void negate_number(void);
void bignum_truncate(void);
void mp_numerator(void);
void mp_denominator(void);
void bignum_power_number(int expo);
double convert_bignum_to_double(unsigned int *p);
double convert_rational_to_double(U *p);
void push_integer(int n);
void push_double(double d);
void push_rational(int a, int b);
int pop_integer(void);
void print_double(U *p, int flag);
void bignum_scan_integer(char *s);
void bignum_scan_float(char *s);
void print_number(U *p);
void gcd_numbers(void);
double pop_double(void);
void bignum_float(void);
void bignum_factorial(int n);
unsigned int * bignum_factorial_nib(int n);
void mp_set_bit(unsigned int *x, unsigned int k);
void mp_clr_bit(unsigned int *x, unsigned int k);
void mshiftright(unsigned int *a);
void eval_binomial(void);
void binomial(void);
void binomial_nib(void);
int binomial_check_args(void);
void eval_ceiling(void);
void ceiling(void);
void yyceiling(void);
void eval_choose(void);
void choose(void);
int choose_check_args(void);
void eval_circexp(void);
void circexp(void);
void clear(void);
void eval_clock(void);
void clockform(void);
void cmdisplay(void);
void eval_coeff(void);
int coeff(void);
void eval_cofactor(void);
void cofactor(U *p, int n, int row, int col);
void eval_condense(void);
void Condense(void);
void yycondense(void);
void eval_conj(void);
void conjugate(void);
void cons(void);
void eval_contract(void);
void contract(void);
void yycontract(void);
void eval_cos(void);
void cosine(void);
void cosine_of_angle_sum(void);
void cosine_of_angle(void);
void eval_cosh(void);
void ycosh(void);
void yycosh(void);
void eval_decomp(void);
void decomp_nib(void);
void decomp_sum(void);
void decomp_product(void);
void define_user_function(void);
void eval_defint(void);
void eval_degree(void);
void degree(void);
void yydegree(U *p);
void eval_denominator(void);
void denominator(void);
void eval_derivative(void);
void derivative(void);
void d_scalar_scalar(void);
void d_scalar_scalar_1(void);
void dsum(void);
void dproduct(void);
void dpower(void);
void dlog(void);
void dd(void);
void dfunction(void);
void dsin(void);
void dcos(void);
void dtan(void);
void darcsin(void);
void darccos(void);
void darctan(void);
void dsinh(void);
void dcosh(void);
void dtanh(void);
void darcsinh(void);
void darccosh(void);
void darctanh(void);
void dabs(void);
void dhermite(void);
void derf(void);
void derfc(void);
void dbesselj0(void);
void dbesseljn(void);
void dbessely0(void);
void dbesselyn(void);
void derivative_of_integral(void);
int det_check_arg(void);
void det(void);
void determinant(int n);
void detg(void);
void yydetg(void);
void lu_decomp(int n);
void display(void);
void emit_top_expr(U *p);
int will_be_displayed_as_fraction(U *p);
void emit_expr(U *p);
void emit_unsigned_expr(U *p);
int is_negative(U *p);
void emit_term(U *p);
int isdenominator(U *p);
int count_denominators(U *p);
void emit_multiply(U *p, int n);
void emit_fraction(U *p, int d);
void emit_numerators(U *p);
void emit_denominators(U *p);
void emit_factor(U *p);
void emit_numerical_fraction(U *p);
int isfactor(U *p);
void emit_power(U *p);
void emit_denominator(U *p, int n);
void emit_function(U *p);
void emit_index_function(U *p);
void emit_factorial_function(U *p);
void emit_subexpr(U *p);
void emit_symbol(U *p);
void emit_string(U *p);
void fixup_fraction(int x, int k1, int k2);
void fixup_power(int k1, int k2);
void move(int j, int k, int dx, int dy);
void get_size(int j, int k, int *h, int *w, int *y);
void displaychar(int c);
void emit_char(int c);
void emit_str(char *s);
void emit_number(U *p, int emit_sign);
int display_cmp(const void *aa, const void *bb);
void print_it(void);
char * getdisplaystr(void);
void fill_buf(void);
void emit_tensor(U *p);
void emit_flat_tensor(U *p);
void emit_tensor_inner(U *p, int j, int *k);
void distill(void);
void distill_nib(void);
void distill_sum(void);
void distill_product(void);
void divisors(void);
void divisors_onstack(void);
void gen(int h, int k);
void factor_add(void);
int divisors_cmp(const void *p1, const void *p2);
void dpow(void);
void eval_draw(void);
void draw_main(void);
void check_for_parametric_draw(void);
void create_point_set(void);
void new_point(double t);
void get_xy(double t);
void eval_f(double t);
void fill(int i, int j, int level);
void setup_trange(void);
void setup_trange_f(void);
void setup_xrange(void);
void setup_xrange_f(void);
void setup_yrange(void);
void setup_yrange_f(void);
void emit_graph(void);
void eval_eigen(void);
void eval_eigenval(void);
void eval_eigenvec(void);
int eigen_check_arg(void);
void eigen(int op);
int step(void);
void step2(int p, int q);
void eval_erf(void);
void yerf(void);
void yyerf(void);
void eval_erfc(void);
void yerfc(void);
void yyerfc(void);
void eval(void);
void eval_sym(void);
void eval_cons(void);
void eval_binding(void);
void eval_check(void);
void eval_det(void);
void eval_dim(void);
void eval_divisors(void);
void eval_do(void);
void eval_eval(void);
void eval_exp(void);
void eval_factorial(void);
void eval_factorpoly(void);
void eval_hermite(void);
void eval_hilbert(void);
void eval_index(void);
void eval_inv(void);
void eval_invg(void);
void eval_isinteger(void);
void eval_multiply(void);
void eval_number(void);
void eval_operator(void);
void eval_print(void);
void eval_quote(void);
void eval_rank(void);
void setq_indexed(void);
void eval_setq(void);
void eval_sqrt(void);
void eval_stop(void);
void eval_subst(void);
void eval_unit(void);
void eval_noexpand(void);
void eval_predicate(void);
void eval_and_print_result(int update);
void eval_expand(void);
void expand(void);
void expand_tensor(void);
void remove_negative_exponents(void);
void expand_get_C(void);
void expand_get_CF(void);
void trivial_divide(void);
void expand_get_B(void);
void expand_get_A(void);
void expand_get_AF(void);
void eval_expcos(void);
void expcos(void);
void eval_expsin(void);
void expsin(void);
void eval_factor(void);
void factor_again(void);
void factor_term(void);
void factor(void);
void factor_small_number(void);
void factorial(void);
void simplifyfactorials(void);
void sfac_product(void);
void sfac_product_f(U **s, int a, int b);
void factorpoly(void);
void yyfactorpoly(void);
void rationalize_coefficients(int h);
int get_factor(void);
void yydivpoly(void);
void evalpoly(void);
int factors(U *p);
void push_term_factors(U *p);
void eval_filter(void);
void filter(void);
void filter_main(void);
void filter_sum(void);
void filter_tensor(void);
int find(U *p, U *q);
void eval_float(void);
void yyfloat(void);
void eval_floor(void);
void yfloor(void);
void yyfloor(void);
void init_font(void);
void draw_text(int font, int x, int y, char *s, int len, int color);
int text_width(int font, char *s);
void get_height_width(int *h, int *w, int font, char *s);
void draw_line(int x1, int y1, int x2, int y2);
void draw_left_bracket(int x, int y, int w, int h);
void draw_right_bracket(int x, int y, int w, int h);
void draw_point(int x, int dx, int y, int dy);
void draw_box(int x1, int y1, int x2, int y2);
void draw_hrule(int x, int y, int w);
void draw_selection_rect(float x, float y, float width, float height);
void eval_for(void);
void eval_gcd(void);
void gcd(void);
void gcd_main(void);
void gcd_expr_expr(void);
void gcd_expr(U *p);
void gcd_term_term(void);
void gcd_term_factor(void);
void gcd_factor_term(void);
void guess(void);
void hermite(void);
void yyhermite(void);
void yyhermite2(int n);
void hilbert(void);
void eval_imag(void);
void imag(void);
void index_function(int n);
void set_component(int n);
void init(void);
void init_symbol_table(void);
void eval_inner(void);
void inner(void);
void inner_f(void);
void eval_integral(void);
void integral(void);
void integral_of_sum(void);
void integral_of_product(void);
void integral_of_form(void);
int inv_check_arg(void);
void inv(void);
void invg(void);
void yyinvg(void);
void inv_decomp(int n);
int iszero(U *p);
int isnegativenumber(U *p);
int isplusone(U *p);
int isminusone(U *p);
int isinteger(U *p);
int isnonnegativeinteger(U *p);
int isposint(U *p);
int ispoly(U *p, U *x);
int ispoly_expr(U *p, U *x);
int ispoly_term(U *p, U *x);
int ispoly_factor(U *p, U *x);
int isnegativeterm(U *p);
int isimaginarynumber(U *p);
int iscomplexnumber(U *p);
int iseveninteger(U *p);
int isnegative(U *p);
int issymbolic(U *p);
int isintegerfactor(U *p);
int isoneover(U *p);
int isfraction(U *p);
int equaln(U *p, int n);
int equalq(U *p, int a, int b);
int isoneoversqrttwo(U *p);
int isminusoneoversqrttwo(U *p);
int isfloating(U *p);
int isimaginaryunit(U *p);
int isquarterturn(U *p);
int isnpi(U *p);
void eval_isprime(void);
void eval_laguerre(void);
void laguerre(void);
void laguerre2(int n);
void eval_lcm(void);
void lcm(void);
void yylcm(void);
void eval_leading(void);
void leading(void);
void eval_legendre(void);
void legendre(void);
void legendre_nib(void);
void legendre2(int n, int m);
void legendre3(int m);
void list(int n);
void eval_log(void);
void logarithm(void);
void yylog(void);
unsigned int * madd(unsigned int *a, unsigned int *b);
unsigned int * msub(unsigned int *a, unsigned int *b);
unsigned int * madd_nib(unsigned int *a, unsigned int *b);
unsigned int * msub_nib(unsigned int *a, unsigned int *b);
int add_ucmp(unsigned int *a, unsigned int *b);
void eval_mag(void);
void mag(void);
void yymag(void);
int main(int argc, char *argv[]);
void run_script(char *filename);
void printstr(char *s);
void printchar(int c);
void printchar_nowrap(int c);
void eval_draw(void);
void eval_sample(void);
void clear_display(void);
void cmdisplay(void);
int mcmp(unsigned int *a, unsigned int *b);
int mcmpint(unsigned int *a, int n);
unsigned int * mgcd(unsigned int *u, unsigned int *v);
void new_string(char *s);
void out_of_memory(void);
void push_zero_matrix(int i, int j);
void push_identity_matrix(int n);
void push_cars(U *p);
void peek(void);
void peek2(void);
int equal(U *p1, U *p2);
int lessp(U *p1, U *p2);
int sign(int n);
int cmp_expr(U *p1, U *p2);
int length(U *p);
U * unique(U *p);
void unique_f(U *p);
void ssqrt(void);
void yyexpand(void);
void exponential(void);
void square(void);
int sort_stack_cmp(const void *p1, const void *p2);
void sort_stack(int n);
unsigned int * mmodpow(unsigned int *x, unsigned int *n, unsigned int *m);
unsigned int * mmul(unsigned int *a, unsigned int *b);
unsigned int * mdiv(unsigned int *a, unsigned int *b);
void addf(unsigned int *a, unsigned int *b, int len);
void subf(unsigned int *a, unsigned int *b, int len);
void mulf(unsigned int *a, unsigned int *b, int len, unsigned int c);
unsigned int * mmod(unsigned int *a, unsigned int *b);
void mdivrem(unsigned int **q, unsigned int **r, unsigned int *a, unsigned int *b);
void eval_mod(void);
void mod(void);
unsigned int * mpow(unsigned int *a, unsigned int n);
int mprime(unsigned int *n);
int mprimef(unsigned int *n, unsigned int *q, int k);
unsigned int * mroot(unsigned int *n, unsigned int index);
unsigned int * mscan(char *s);
unsigned int * maddf(unsigned int *a, int n);
unsigned int * mmulf(unsigned int *a, int n);
char * mstr(unsigned int *a);
int divby1billion(unsigned int *a);
void multiply(void);
void yymultiply(void);
void parse_p1(void);
void parse_p2(void);
void combine_factors(int h);
void multiply_noexpand(void);
void multiply_all(int n);
void multiply_all_noexpand(int n);
void divide(void);
void inverse(void);
void reciprocate(void);
void negate(void);
void negate_expand(void);
void negate_noexpand(void);
void normalize_radical_factors(int h);
int is_radical_number(U *p);
void eval_nroots(void);
void monic(int n);
void findroot(int n);
void compute_fa(int n);
void divpoly_FIXME(int n);
void eval_numerator(void);
void numerator(void);
void eval_outer(void);
void outer(void);
void yyouter(void);
void partition(void);
void eval_polar(void);
void polar(void);
void factor_number(void);
void factor_a(void);
void try_kth_prime(int k);
int factor_b(void);
void push_factor(unsigned int *d, int count);
void eval_power(void);
void power(void);
void yypower(void);
int simplify_polar(void);
void eval_prime(void);
void prime(void);
void print(U *p);
void print_subexpr(U *p);
void print_expr(U *p);
int sign_of_term(U *p);
void print_a_over_b(U *p);
void print_term(U *p);
void print_denom(U *p, int d);
void print_factor(U *p);
void print_index_function(U *p);
void print_factorial_function(U *p);
void print_tensor(U *p);
void print_tensor_inner(U *p, int j, int *k);
void print_str(char *s);
void print_char(int c);
void print_function_definition(U *p);
void print_arg_list(U *p);
void print_lisp(U *p);
void print1(U *p);
void print_multiply_sign(void);
int is_denominator(U *p);
int any_denominators(U *p);
void eval_product(void);
void qadd(void);
void qdiv(void);
void qmul(void);
void qpow(void);
void qpowf(void);
void normalize_angle(void);
int is_small_integer(U *p);
void qsub(void);
void quickfactor(void);
void quickpower(void);
void eval_quotient(void);
void divpoly(void);
void eval_rationalize(void);
void rationalize(void);
void yyrationalize(void);
void multiply_denominators(U *p);
void multiply_denominators_term(U *p);
void multiply_denominators_factor(U *p);
void rationalize_tensor(void);
void rationalize_lcm(void);
void eval_real(void);
void real(void);
void eval_rect(void);
void rect(void);
void rewrite(void);
void rewrite_tensor(void);
void eval_roots(void);
void roots(void);
void roots2(void);
void roots3(void);
void mini_solve(void);
void run_as_thread(char *s);
void * run1(void *s);
void run(char *s);
void check_stack(void);
void echo_input(char *s);
void check_esc_flag(void);
void stop(char *s);
int scan(char *s);
int scan_meta(char *s);
void scan_stmt(void);
void scan_relation(void);
void scan_expression(void);
int is_factor(void);
void scan_term(void);
void scan_power(void);
void scan_factor(void);
void scan_symbol(void);
void scan_string(void);
void scan_function_call(void);
void scan_subexpr(void);
void error(char *errmsg);
void build_tensor(int n);
void get_next_token(void);
void get_token(void);
void update_token_buf(char *a, char *b);
void test_madd(void);
void test_maddf(int na, int nb, int nc);
void test_msub(void);
void test_msubf(int na, int nb, int nc);
void test_mcmp(void);
void test_mgcd(void);
unsigned int * egcd(unsigned int *a, unsigned int *b);
void test_mmodpow(void);
void test_mmul(void);
void test_mmulf(int na, int nb, int nc);
void test_mdiv(void);
void test_mdivf(int na, int nb, int nc);
void test_mmod(void);
void test_mmodf(int na, int nb, int nc);
void test_mpow(void);
void test_mroot(void);
void test_quickfactor(void);
void test_all(void);
void eval_sgn(void);
void sgn(void);
void eval_simfac(void);
void simfac(void);
void simfac(void);
void simfac_term(void);
int yysimfac(int h);
void eval_simplify(void);
void simplify(void);
void simplify_main(void);
void simplify_tensor(void);
int count(U *p);
void f1(void);
void f2(void);
void f3(void);
void f4(void);
void simplify_trig(void);
void f5(void);
void f9(void);
int nterms(U *p);
void eval_sin(void);
void sine(void);
void sine_of_angle_sum(void);
void sine_of_angle(void);
void eval_sinh(void);
void ysinh(void);
void yysinh(void);
void push(U *p);
U * pop(void);
void push_frame(int n);
void pop_frame(int n);
void save(void);
void restore(void);
void swap(void);
void dupl(void);
void subst(void);
void eval_sum(void);
void std_symbol(char *s, int n);
U * usr_symbol(char *s);
char * get_printname(U *p);
void set_binding(U *p, U *b);
void set_binding_and_arglist(U *p, U *b, U *a);
U * get_binding(U *p);
U * get_arglist(U *p);
int symnum(U *p);
void push_binding(U *p);
void pop_binding(U *p);
void eval_tan(void);
void tangent(void);
void yytangent(void);
void eval_tanh(void);
void eval_taylor(void);
void taylor(void);
void eval_tensor(void);
void tensor_plus_tensor(void);
void tensor_times_scalar(void);
void scalar_times_tensor(void);
int is_square_matrix(U *p);
void d_tensor_tensor(void);
void d_scalar_tensor(void);
void d_tensor_scalar(void);
int compare_tensors(U *p1, U *p2);
void power_tensor(void);
void copy_tensor(void);
void promote_tensor(void);
int compatible(U *p, U *q);
void eval_test(void);
void eval_testeq(void);
void eval_testge(void);
void eval_testgt(void);
void eval_testle(void);
void eval_testlt(void);
void eval_not(void);
void eval_and(void);
void eval_or(void);
int cmp_args(void);
void transform(char **s);
int f_equals_a(int h);
void eval_transpose(void);
void transpose(void);
void eval_user_function(void);
int rewrite_args(void);
int rewrite_args_tensor(void);
void variables(void);
void lscan(U *p);
int var_cmp(const void *p1, const void *p2);
void vectorize(int n);
void printchar(int c);
void printchar_nowrap(int c);
void printstr(char *s);
void shipout(struct display *p);
void clear_display(void);
int check_display(void);
void get_view(int *h, int *w);
void draw_display(int y1, int y2);
void eval_zero(void);

// Absolute value, aka vector magnitude

void
eval_abs(void)
{
	push(cadr(p1));
	eval();
	absval();
}

void
absval(void)
{
	int h;
	save();
	p1 = pop();
	if (istensor(p1)) {
		absval_tensor();
		restore();
		return;
	}
	if (isnum(p1)) {
		push(p1);
		if (isnegativenumber(p1))
			negate();
		restore();
		return;
	}
	if (iscomplexnumber(p1)) {
		push(p1);
		push(p1);
		conjugate();
		multiply();
		push_rational(1, 2);
		power();
		restore();
		return;
	}
	// abs(1/a) evaluates to 1/abs(a)
	if (car(p1) == symbol(POWER) && isnegativeterm(caddr(p1))) {
		push(p1);
		reciprocate();
		absval();
		reciprocate();
		restore();
		return;
	}
	// abs(a*b) evaluates to abs(a)*abs(b)
	if (car(p1) == symbol(MULTIPLY)) {
		h = tos;
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			absval();
			p1 = cdr(p1);
		}
		multiply_all(tos - h);
		restore();
		return;
	}
	if (isnegativeterm(p1) || (car(p1) == symbol(ADD) && isnegativeterm(cadr(p1)))) {
		push(p1);
		negate();
		p1 = pop();
	}
	push_symbol(ABS);
	push(p1);
	list(2);
	restore();
}

void
absval_tensor(void)
{
	if (p1->u.tensor->ndim != 1)
		stop("abs(tensor) with tensor rank > 1");
	push(p1);
	push(p1);
	conjugate();
	inner();
	push_rational(1, 2);
	power();
	simplify();
	eval();
}

/* Symbolic addition

	Terms in a sum are combined if they are identical modulo rational
	coefficients.

	For example, A + 2A becomes 3A.

	However, the sum A + sqrt(2) A is not modified.

	Combining terms can lead to second-order effects.

	For example, consider the case of

		1/sqrt(2) A + 3/sqrt(2) A + sqrt(2) A

	The first two terms are combined to yield 2 sqrt(2) A.

	This result can now be combined with the third term to yield

		3 sqrt(2) A
*/

int flag;

void
eval_add(void)
{
	int h = tos;
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		p2 = pop();
		push_terms(p2);
		p1 = cdr(p1);
	}
	add_terms(tos - h);
}

/* Add n terms, returns one expression on the stack. */

void
add_terms(int n)
{
	int i, h;
	U **s;
	h = tos - n;
	s = stack + h;
	/* ensure no infinite loop, use "for" */
	for (i = 0; i < 10; i++) {
		if (n < 2)
			break;
		flag = 0;
		qsort(s, n, sizeof (U *), cmp_terms);
		if (flag == 0)
			break;
		n = combine_terms(s, n);
	}
	tos = h + n;
	switch (n) {
	case 0:
		push_integer(0);
		break;
	case 1:
		break;
	default:
		list(n);
		p1 = pop();
		push_symbol(ADD);
		push(p1);
		cons();
		break;
	}
}

/* Compare terms for order, clobbers p1 and p2. */

int
cmp_terms(const void *q1, const void *q2)
{
	int i, t;
	p1 = *((U **) q1);
	p2 = *((U **) q2);
	/* numbers can be combined */
	if (isnum(p1) && isnum(p2)) {
		flag = 1;
		return 0;
	}
	/* congruent tensors can be combined */
	if (istensor(p1) && istensor(p2)) {
		if (p1->u.tensor->ndim < p2->u.tensor->ndim)
			return -1;
		if (p1->u.tensor->ndim > p2->u.tensor->ndim)
			return 1;
		for (i = 0; i < p1->u.tensor->ndim; i++) {
			if (p1->u.tensor->dim[i] < p2->u.tensor->dim[i])
				return -1;
			if (p1->u.tensor->dim[i] > p2->u.tensor->dim[i])
				return 1;
		}
		flag = 1;
		return 0;
	}
	if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		if (isnum(car(p1))) {
			p1 = cdr(p1);
			if (cdr(p1) == symbol(NIL))
				p1 = car(p1);
		}
	}
	if (car(p2) == symbol(MULTIPLY)) {
		p2 = cdr(p2);
		if (isnum(car(p2))) {
			p2 = cdr(p2);
			if (cdr(p2) == symbol(NIL))
				p2 = car(p2);
		}
	}
	t = cmp_expr(p1, p2);
	if (t == 0)
		flag = 1;
	return t;
}

/* Compare adjacent terms in s[] and combine if possible.

	Returns the number of terms remaining in s[].

	n	number of terms in s[] initially
*/

int
combine_terms(U **s, int n)
{
	int i, j, t;
	for (i = 0; i < n - 1; i++) {
		check_esc_flag();
		p3 = s[i];
		p4 = s[i + 1];
		if (istensor(p3) && istensor(p4)) {
			push(p3);
			push(p4);
			tensor_plus_tensor();
			p1 = pop();
			if (p1 != symbol(NIL)) {
				s[i] = p1;
				for (j = i + 1; j < n - 1; j++)
					s[j] = s[j + 1];
				n--;
				i--;
			}
			continue;
		}
		if (istensor(p3) || istensor(p4))
			continue;
		if (isnum(p3) && isnum(p4)) {
			push(p3);
			push(p4);
			add_numbers();
			p1 = pop();
			if (iszero(p1)) {
				for (j = i; j < n - 2; j++)
					s[j] = s[j + 2];
				n -= 2;
			} else {
				s[i] = p1;
				for (j = i + 1; j < n - 1; j++)
					s[j] = s[j + 1];
				n--;
			}
			i--;
			continue;
		}
		if (isnum(p3) || isnum(p4))
			continue;
		p1 = one;
		p2 = one;
		t = 0;
		if (car(p3) == symbol(MULTIPLY)) {
			p3 = cdr(p3);
			t = 1; /* p3 is now denormal */
			if (isnum(car(p3))) {
				p1 = car(p3);
				p3 = cdr(p3);
				if (cdr(p3) == symbol(NIL)) {
					p3 = car(p3);
					t = 0;
				}
			}
		}
		if (car(p4) == symbol(MULTIPLY)) {
			p4 = cdr(p4);
			if (isnum(car(p4))) {
				p2 = car(p4);
				p4 = cdr(p4);
				if (cdr(p4) == symbol(NIL))
					p4 = car(p4);
			}
		}
		if (!equal(p3, p4))
			continue;
		push(p1);
		push(p2);
		add_numbers();
		p1 = pop();
		if (iszero(p1)) {
			for (j = i; j < n - 2; j++)
				s[j] = s[j + 2];
			n -= 2;
			i--;
			continue;
		}
		push(p1);
		if (t) {
			push(symbol(MULTIPLY));
			push(p3);
			cons();
		} else
			push(p3);
		multiply();
		s[i] = pop();
		for (j = i + 1; j < n - 1; j++)
			s[j] = s[j + 1];
		n--;
		i--;
	}
	return n;
}

void
push_terms(U *p)
{
	if (car(p) == symbol(ADD)) {
		p = cdr(p);
		while (iscons(p)) {
			push(car(p));
			p = cdr(p);
		}
	} else if (!iszero(p))
		push(p);
}

/* add two expressions */

void
add(void)
{
	int h;
	save();
	p2 = pop();
	p1 = pop();
	h = tos;
	push_terms(p1);
	push_terms(p2);
	add_terms(tos - h);
	restore();
}

void
add_all(int k)
{
	int h, i;
	U **s;
	save();
	s = stack + tos - k;
	h = tos;
	for (i = 0; i < k; i++)
		push_terms(s[i]);
	add_terms(tos - h);
	p1 = pop();
	tos -= k;
	push(p1);
	restore();
}

void
subtract(void)
{
	negate();
	add();
}

// Adjunct of a matrix

void
eval_adj(void)
{
	push(cadr(p1));
	eval();
	adj();
}

void
adj(void)
{
	int i, j, n;
	save();
	p1 = pop();
	if (istensor(p1) && p1->u.tensor->ndim == 2 && p1->u.tensor->dim[0] == p1->u.tensor->dim[1])
		;
	else
		stop("adj: square matrix expected");
	n = p1->u.tensor->dim[0];
	p2 = alloc_tensor(n * n);
	p2->u.tensor->ndim = 2;
	p2->u.tensor->dim[0] = n;
	p2->u.tensor->dim[1] = n;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {
			cofactor(p1, n, i, j);
			p2->u.tensor->elem[n * j + i] = pop(); /* transpose */
		}
	push(p2);
	restore();
}

#undef M
#undef N

#define M 1000		// maximum M blocks
#define N 100000	// N atoms per block

U *mem[M];
int mcount;

U *free_list;
int free_count;

U *
alloc(void)
{
	U *p;
	if (free_count == 0) {
		if (mcount == 0)
			alloc_mem();
		else {
			gc();
			if (free_count < N * mcount / 2)
				alloc_mem();
		}
		if (free_count == 0)
			stop("atom space exhausted");
	}
	p = free_list;
	free_list = free_list->u.cons.cdr;
	free_count--;
	return p;
}

U *
alloc_tensor(int nelem)
{
	int i;
	U *p;
	p = alloc();
	p->k = TENSOR;
	p->u.tensor = (T *) malloc(sizeof (T) + nelem * sizeof (U *));
	if (p->u.tensor == NULL)
		out_of_memory();
	p->u.tensor->nelem = nelem;
	for (i = 0; i < nelem; i++)
		p->u.tensor->elem[i] = zero;
	return p;
}

// garbage collector

void
gc(void)
{
	int i, j;
	U *p;
	// tag everything
	for (i = 0; i < mcount; i++) {
		p = mem[i];
		for (j = 0; j < N; j++)
			p[j].tag = 1;
	}
	// untag what's used
	untag(p0);
	untag(p1);
	untag(p2);
	untag(p3);
	untag(p4);
	untag(p5);
	untag(p6);
	untag(p7);
	untag(p8);
	untag(p9);
	for (i = 0; i < NSYM; i++) {
		untag(binding[i]);
		untag(arglist[i]);
	}
	for (i = 0; i < tos; i++)
		untag(stack[i]);
	for (i = (int) (frame - stack); i < TOS; i++)
		untag(stack[i]);
	// collect everything that's still tagged
	free_count = 0;
	for (i = 0; i < mcount; i++) {
		p = mem[i];
		for (j = 0; j < N; j++) {
			if (p[j].tag == 0)
				continue;
			// still tagged so it's unused, put on free list
			switch (p[j].k) {
			case TENSOR:
				free(p[j].u.tensor);
				break;
			case STR:
				free(p[j].u.str);
				break;
			case NUM:
				mfree(p[j].u.q.a);
				mfree(p[j].u.q.b);
				break;
			}
			p[j].k = CONS; // so no double free occurs above
			p[j].u.cons.cdr = free_list;
			free_list = p + j;
			free_count++;
		}
	}
}

void
untag(U *p)
{
	int i;
	if (iscons(p)) {
		do {
			if (p->tag == 0)
				return;
			p->tag = 0;
			untag(p->u.cons.car);
			p = p->u.cons.cdr;
		} while (iscons(p));
		untag(p);
		return;
	}
	if (p->tag) {
		p->tag = 0;
		if (istensor(p)) {
			for (i = 0; i < p->u.tensor->nelem; i++)
				untag(p->u.tensor->elem[i]);
		}
	}
}

// get memory for N atoms

void
alloc_mem(void)
{
	int i;
	U *p;
	if (mcount == M)
		return;
	p = (U *) malloc(N * sizeof (U));
	if (p == NULL)
		return;
	mem[mcount++] = p;
	for (i = 0; i < N; i++) {
		p[i].k = CONS; // so no free in gc
		p[i].u.cons.cdr = p + i + 1;
	}
	p[N - 1].u.cons.cdr = free_list;
	free_list = p;
	free_count += N;
}

void
print_mem_info(void)
{
	char buf[100];
	sprintf(buf, "%d blocks (%d bytes/block)\n", N * mcount, (int) sizeof (U));
	printstr(buf);
	sprintf(buf, "%d free\n", free_count);
	printstr(buf);
	sprintf(buf, "%d used\n", N * mcount - free_count);
	printstr(buf);
}

// Append one list to another.

void
append(void)
{
	int h;
	save();
	p2 = pop();
	p1 = pop();
	h = tos;
	if (iscons(p1))
		while (iscons(p1)) {
			push(car(p1));
			p1 = cdr(p1);
		}
	else
		push(p1);
	if (iscons(p2))
		while (iscons(p2)) {
			push(car(p2));
			p2 = cdr(p2);
		}
	else
		push(p2);
	list(tos - h);
	restore();
}

void
eval_arccos(void)
{
	push(cadr(p1));
	eval();
	arccos();
}

void
arccos(void)
{
	int n;
	double d;
	save();
	p1 = pop();
	if (car(p1) == symbol(COS)) {
		push(cadr(p1));
		restore();
		return;
	}
	if (isdouble(p1)) {
		errno = 0;
		d = acos(p1->u.d);
		if (errno)
			stop("arccos function argument is not in the interval [-1,1]");
		push_double(d);
		restore();
		return;
	}
	// if p1 == 1/sqrt(2) then return 1/4*pi (45 degrees)
	if (isoneoversqrttwo(p1)) {
		push_rational(1, 4);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	// if p1 == -1/sqrt(2) then return 3/4*pi (135 degrees)
	if (isminusoneoversqrttwo(p1)) {
		push_rational(3, 4);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	if (!isrational(p1)) {
		push_symbol(ARCCOS);
		push(p1);
		list(2);
		restore();
		return;
	}
	push(p1);
	push_integer(2);
	multiply();
	n = pop_integer();
	switch (n) {
	case -2:
		push_symbol(PI);
		break;
	case -1:
		push_rational(2, 3);
		push_symbol(PI);
		multiply();
		break;
	case 0:
		push_rational(1, 2);
		push_symbol(PI);
		multiply();
		break;
	case 1:
		push_rational(1, 3);
		push_symbol(PI);
		multiply();
		break;
	case 2:
		push(zero);
		break;
	default:
		push_symbol(ARCCOS);
		push(p1);
		list(2);
		break;
	}
	restore();
}

void
eval_arccosh(void)
{
	push(cadr(p1));
	eval();
	arccosh();
}

void
arccosh(void)
{
	double d;
	save();
	p1 = pop();
	if (car(p1) == symbol(COSH)) {
		push(cadr(p1));
		restore();
		return;
	}
	if (isdouble(p1)) {
		d = p1->u.d;
		if (d < 1.0)
			stop("arccosh function argument is less than 1.0");
		d = log(d + sqrt(d * d - 1.0));
		push_double(d);
		restore();
		return;
	}
	if (isplusone(p1)) {
		push(zero);
		restore();
		return;
	}
	push_symbol(ARCCOSH);
	push(p1);
	list(2);
	restore();
}

void
eval_arcsin(void)
{
	push(cadr(p1));
	eval();
	arcsin();
}

void
arcsin(void)
{
	int n;
	double d;
	save();
	p1 = pop();
	if (car(p1) == symbol(SIN)) {
		push(cadr(p1));
		restore();
		return;
	}
	if (isdouble(p1)) {
		errno = 0;
		d = asin(p1->u.d);
		if (errno)
			stop("arcsin function argument is not in the interval [-1,1]");
		push_double(d);
		restore();
		return;
	}
	// if p1 == 1/sqrt(2) then return 1/4*pi (45 degrees)
	if (isoneoversqrttwo(p1)) {
		push_rational(1, 4);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	// if p1 == -1/sqrt(2) then return -1/4*pi (-45 degrees)
	if (isminusoneoversqrttwo(p1)) {
		push_rational(-1, 4);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	if (!isrational(p1)) {
		push_symbol(ARCSIN);
		push(p1);
		list(2);
		restore();
		return;
	}
	push(p1);
	push_integer(2);
	multiply();
	n = pop_integer();
	switch (n) {
	case -2:
		push_rational(-1, 2);
		push_symbol(PI);
		multiply();
		break;
	case -1:
		push_rational(-1, 6);
		push_symbol(PI);
		multiply();
		break;
	case 0:
		push(zero);
		break;
	case 1:
		push_rational(1, 6);
		push_symbol(PI);
		multiply();
		break;
	case 2:
		push_rational(1, 2);
		push_symbol(PI);
		multiply();
		break;
	default:
		push_symbol(ARCSIN);
		push(p1);
		list(2);
		break;
	}
	restore();
}

void
eval_arcsinh(void)
{
	push(cadr(p1));
	eval();
	arcsinh();
}

void
arcsinh(void)
{
	double d;
	save();
	p1 = pop();
	if (car(p1) == symbol(SINH)) {
		push(cadr(p1));
		restore();
		return;
	}
	if (isdouble(p1)) {
		d = p1->u.d;
		d = log(d + sqrt(d * d + 1.0));
		push_double(d);
		restore();
		return;
	}
	if (iszero(p1)) {
		push(zero);
		restore();
		return;
	}
	push_symbol(ARCSINH);
	push(p1);
	list(2);
	restore();
}

void
eval_arctan(void)
{
	push(cadr(p1));
	eval();
	arctan();
}

void
arctan(void)
{
	double d;
	save();
	p1 = pop();
	if (car(p1) == symbol(TAN)) {
		push(cadr(p1));
		restore();
		return;
	}
	if (isdouble(p1)) {
		errno = 0;
		d = atan(p1->u.d);
		if (errno)
			stop("arctan function error");
		push_double(d);
		restore();
		return;
	}
	if (iszero(p1)) {
		push(zero);
		restore();
		return;
	}
	if (isnegative(p1)) {
		push(p1);
		negate();
		arctan();
		negate();
		restore();
		return;
	}
	// arctan(sin(a) / cos(a)) ?
	if (find(p1, symbol(SIN)) && find(p1, symbol(COS))) {
		push(p1);
		numerator();
		p2 = pop();
		push(p1);
		denominator();
		p3 = pop();
		if (car(p2) == symbol(SIN) && car(p3) == symbol(COS) && equal(cadr(p2), cadr(p3))) {
			push(cadr(p2));
			restore();
			return;
		}
	}
	// arctan(1/sqrt(3)) -> pi/6
	if (car(p1) == symbol(POWER) && equaln(cadr(p1), 3) && equalq(caddr(p1), -1, 2)) {
		push_rational(1, 6);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	// arctan(1) -> pi/4
	if (equaln(p1, 1)) {
		push_rational(1, 4);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	// arctan(sqrt(3)) -> pi/3
	if (car(p1) == symbol(POWER) && equaln(cadr(p1), 3) && equalq(caddr(p1), 1, 2)) {
		push_rational(1, 3);
		push_symbol(PI);
		multiply();
		restore();
		return;
	}
	push_symbol(ARCTAN);
	push(p1);
	list(2);
	restore();
}

void
eval_arctanh(void)
{
	push(cadr(p1));
	eval();
	arctanh();
}

void
arctanh(void)
{
	double d;
	save();
	p1 = pop();
	if (car(p1) == symbol(TANH)) {
		push(cadr(p1));
		restore();
		return;
	}
	if (isdouble(p1)) {
		d = p1->u.d;
		if (d < -1.0 || d > 1.0)
			stop("arctanh function argument is not in the interval [-1,1]");
		d = log((1.0 + d) / (1.0 - d)) / 2.0;
		push_double(d);
		restore();
		return;
	}
	if (iszero(p1)) {
		push(zero);
		restore();
		return;
	}
	push_symbol(ARCTANH);
	push(p1);
	list(2);
	restore();
}

/* Argument (angle) of complex z

	z		arg(z)
	-		------

	a		0

	-a		-pi			See note 3 below

	(-1)^a		a pi

	exp(a + i b)	b

	a b		arg(a) + arg(b)

	a + i b		arctan(b/a)

Result by quadrant

	z		arg(z)
	-		------

	1 + i		1/4 pi

	1 - i		-1/4 pi

	-1 + i		3/4 pi

	-1 - i		-3/4 pi

Notes

	1. Handles mixed polar and rectangular forms, e.g. 1 + exp(i pi/3)

	2. Symbols in z are assumed to be positive and real.

	3. Negative direction adds -pi to angle.

	Example: z = (-1)^(1/3), mag(z) = 1/3 pi, mag(-z) = -2/3 pi
*/

void
eval_arg(void)
{
	push(cadr(p1));
	eval();
	arg();
}

void
arg(void)
{
	save();
	p1 = pop();
	push(p1);
	numerator();
	yyarg();
	push(p1);
	denominator();
	yyarg();
	subtract();
	restore();
}

#undef RE
#undef IM

#define RE p2
#define IM p3

void
yyarg(void)
{
	save();
	p1 = pop();
	if (isnegativenumber(p1)) {
		push(symbol(PI));
		negate();
	} else if (car(p1) == symbol(POWER) && equaln(cadr(p1), -1)) {
		// -1 to a power
		push(symbol(PI));
		push(caddr(p1));
		multiply();
	} else if (car(p1) == symbol(POWER) && cadr(p1) == symbol(EXP1)) {
		// exponential
		push(caddr(p1));
		imag();
	} else if (car(p1) == symbol(MULTIPLY)) {
		// product of factors
		push_integer(0);
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			arg();
			add();
			p1 = cdr(p1);
		}
	} else if (car(p1) == symbol(ADD)) {
		// sum of terms
		push(p1);
		rect();
		p1 = pop();
		push(p1);
		real();
		RE = pop();
		push(p1);
		imag();
		IM = pop();
		if (iszero(RE)) {
			push(symbol(PI));
			if (isnegative(IM))
				negate();
		} else {
			push(IM);
			push(RE);
			divide();
			arctan();
			if (isnegative(RE)) {
				push_symbol(PI);
				if (isnegative(IM))
					subtract();	// quadrant 1 -> 3
				else
					add();		// quadrant 4 -> 2
			}
		}
	} else
		// pure real
		push_integer(0);
	restore();
}

void
eval_atomize(void)
{
	push(cadr(p1));
	eval();
	p1 = pop();
	if (iscons(p1))
		atomize();
	else
		push(p1);
}

void
atomize(void)
{
	int i, n;
	p1 = cdr(p1);
	n = length(p1);
	if (n == 1) {
		push(car(p1));
		return;
	}
	p2 = alloc_tensor(n);
	p2->u.tensor->ndim = 1;
	p2->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++) {
		p2->u.tensor->elem[i] = car(p1);
		p1 = cdr(p1);
	}
	push(p2);
}

// pretty print

void
bake(void)
{
	int h, s, t, x, y, z;
	expanding++;
	save();
	p1 = pop();
	if (length(p1) > 102) { // too slow for large polynomials
		push(p1);
		restore();
		return;
	}
	s = ispoly(p1, symbol(SYMBOL_S));
	t = ispoly(p1, symbol(SYMBOL_T));
	x = ispoly(p1, symbol(SYMBOL_X));
	y = ispoly(p1, symbol(SYMBOL_Y));
	z = ispoly(p1, symbol(SYMBOL_Z));
	if (s == 1 && t == 0 && x == 0 && y == 0 && z == 0) {
		p2 = symbol(SYMBOL_S);
		bake_poly();
	} else if (s == 0 && t == 1 && x == 0 && y == 0 && z == 0) {
		p2 = symbol(SYMBOL_T);
		bake_poly();
	} else if (s == 0 && t == 0 && x == 1 && y == 0 && z == 0) {
		p2 = symbol(SYMBOL_X);
		bake_poly();
	} else if (s == 0 && t == 0 && x == 0 && y == 1 && z == 0) {
		p2 = symbol(SYMBOL_Y);
		bake_poly();
	} else if (s == 0 && t == 0 && x == 0 && y == 0 && z == 1) {
		p2 = symbol(SYMBOL_Z);
		bake_poly();
	} else if (iscons(p1)) {
		h = tos;
		push(car(p1));
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			bake();
			p1 = cdr(p1);
		}
		list(tos - h);
	} else
		push(p1);
	restore();
	expanding--;
}

void
polyform(void)
{
	int h;
	save();
	p2 = pop();
	p1 = pop();
	if (ispoly(p1, p2))
		bake_poly();
	else if (iscons(p1)) {
		h = tos;
		push(car(p1));
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			push(p2);
			polyform();
			p1 = cdr(p1);
		}
		list(tos - h);
	} else
		push(p1);
	restore();
}

void
bake_poly(void)
{
	int h, i, k, n;
	U **a;
	a = stack + tos;
	push(p1);		// p(x)
	push(p2);		// x
	k = coeff();
	h = tos;
	for (i = k - 1; i >= 0; i--) {
		p1 = a[i];
		bake_poly_term(i);
	}
	n = tos - h;
	if (n > 1) {
		list(n);
		push(symbol(ADD));
		swap();
		cons();
	}
	p1 = pop();
	tos -= k;
	push(p1);
}

// p1 points to coefficient of p2 ^ k

void
bake_poly_term(int k)
{
	int h, n;
	if (iszero(p1))
		return;
	// constant term?
	if (k == 0) {
		if (car(p1) == symbol(ADD)) {
			p1 = cdr(p1);
			while (iscons(p1)) {
				push(car(p1));
				p1 = cdr(p1);
			}
		} else
			push(p1);
		return;
	}
	h = tos;
	// coefficient
	if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			p1 = cdr(p1);
		}
	} else if (!equaln(p1, 1))
		push(p1);
	// x ^ k
	if (k == 1)
		push(p2);
	else {
		push(symbol(POWER));
		push(p2);
		push_integer(k);
		list(3);
	}
	n = tos - h;
	if (n > 1) {
		list(n);
		push(symbol(MULTIPLY));
		swap();
		cons();
	}
}

/* Bessel J function

	1st arg		x

	2nd arg		n

Recurrence relation

	besselj(x,n) = (2/x) (n-1) besselj(x,n-1) - besselj(x,n-2)

	besselj(x,1/2) = sqrt(2/pi/x) sin(x)

	besselj(x,-1/2) = sqrt(2/pi/x) cos(x)

For negative n, reorder the recurrence relation as

	besselj(x,n-2) = (2/x) (n-1) besselj(x,n-1) - besselj(x,n)

Substitute n+2 for n to obtain

	besselj(x,n) = (2/x) (n+1) besselj(x,n+1) - besselj(x,n+2)

Examples

	besselj(x,3/2) = (1/x) besselj(x,1/2) - besselj(x,-1/2)

	besselj(x,-3/2) = -(1/x) besselj(x,-1/2) - besselj(x,1/2)
*/

void
eval_besselj(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	besselj();
}

void
besselj(void)
{
	save();
	yybesselj();
	restore();
}

#undef X
#undef N
#undef SIGN

#define X p1
#define N p2
#define SIGN p3

void
yybesselj(void)
{
	double d;
	int n;
	N = pop();
	X = pop();
	push(N);
	n = pop_integer();
	// numerical result
	if (isdouble(X) && n != (int) 0x80000000) {
		d = jn(n, X->u.d);
		push_double(d);
		return;
	}
	// bessej(0,0) = 1
	if (iszero(X) && iszero(N)) {
		push_integer(1);
		return;
	}
	// besselj(0,n) = 0
	if (iszero(X) && n != (int) 0x80000000) {
		push_integer(0);
		return;
	}
	// half arguments
	if (N->k == NUM && MEQUAL(N->u.q.b, 2)) {
		// n = 1/2
		if (MEQUAL(N->u.q.a, 1)) {
			push_integer(2);
			push_symbol(PI);
			divide();
			push(X);
			divide();
			push_rational(1, 2);
			power();
			push(X);
			sine();
			multiply();
			return;
		}
		// n = -1/2
		if (MEQUAL(N->u.q.a, -1)) {
			push_integer(2);
			push_symbol(PI);
			divide();
			push(X);
			divide();
			push_rational(1, 2);
			power();
			push(X);
			cosine();
			multiply();
			return;
		}
		// besselj(x,n) = (2/x) (n-sgn(n)) besselj(x,n-sgn(n)) - besselj(x,n-2*sgn(n))
		push_integer(MSIGN(N->u.q.a));
		SIGN = pop();
		push_integer(2);
		push(X);
		divide();
		push(N);
		push(SIGN);
		subtract();
		multiply();
		push(X);
		push(N);
		push(SIGN);
		subtract();
		besselj();
		multiply();
		push(X);
		push(N);
		push_integer(2);
		push(SIGN);
		multiply();
		subtract();
		besselj();
		subtract();
		return;
	}
#if 0 // test cases needed
	if (isnegativeterm(X)) {
		push(X);
		negate();
		push(N);
		power();
		push(X);
		push(N);
		negate();
		power();
		multiply();
		push_symbol(BESSELJ);
		push(X);
		negate();
		push(N);
		list(3);
		multiply();
		return;
	}
	if (isnegativeterm(N)) {
		push_integer(-1);
		push(N);
		power();
		push_symbol(BESSELJ);
		push(X);
		push(N);
		negate();
		list(3);
		multiply();
		return;
	}
#endif
	push(symbol(BESSELJ));
	push(X);
	push(N);
	list(3);
}

//-----------------------------------------------------------------------------
//
//	Bessel Y function
//
//	Input:		tos-2		x	(can be a symbol or expr)
//
//			tos-1		n
//
//	Output:		Result on stack
//
//-----------------------------------------------------------------------------

void
eval_bessely(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	bessely();
}

void
bessely(void)
{
	save();
	yybessely();
	restore();
}

#undef X
#undef N

#define X p1
#define N p2

void
yybessely(void)
{
	double d;
	int n;
	N = pop();
	X = pop();
	push(N);
	n = pop_integer();
	if (isdouble(X) && n != (int) 0x80000000) {
		d = yn(n, X->u.d);
		push_double(d);
		return;
	}
	if (isnegativeterm(N)) {
		push_integer(-1);
		push(N);
		power();
		push_symbol(BESSELY);
		push(X);
		push(N);
		negate();
		list(3);
		multiply();
		return;
	}
	push_symbol(BESSELY);
	push(X);
	push(N);
	list(3);
	return;
}

#define MP_MIN_SIZE 2
#define MP_MAX_FREE 1000

int mtotal, mfreecount;
unsigned int *free_stack[MP_MAX_FREE];

unsigned int *
mnew(int n)
{
	unsigned int *p;
	if (n < MP_MIN_SIZE)
		n = MP_MIN_SIZE;
	if (n == MP_MIN_SIZE && mfreecount)
		p = free_stack[--mfreecount];
	else {
		p = (unsigned int *) malloc((n + 3) * sizeof (int));
		if (p == 0)
			stop("malloc failure");
	}
	p[0] = n;
	mtotal += n;
	return p + 3;
}

void
mfree(unsigned int *p)
{
	p -= 3;
	mtotal -= p[0];
	if (p[0] == MP_MIN_SIZE && mfreecount < MP_MAX_FREE)
		free_stack[mfreecount++] = p;
	else
		free(p);
}

// convert int to bignum

unsigned int *
mint(int n)
{
	unsigned int *p = mnew(1);
	if (n < 0)
		MSIGN(p) = -1;
	else
		MSIGN(p) = 1;
	MLENGTH(p) = 1;
	p[0] = abs(n);
	return p;
}

// copy bignum

unsigned int *
mcopy(unsigned int *a)
{
	int i;
	unsigned int *b;
	b = mnew(MLENGTH(a));
	MSIGN(b) = MSIGN(a);
	MLENGTH(b) = MLENGTH(a);
	for (i = 0; i < MLENGTH(a); i++)
		b[i] = a[i];
	return b;
}

// a >= b ?

int
ge(unsigned int *a, unsigned int *b, int len)
{
	int i;
	for (i = len - 1; i > 0; i--)
		if (a[i] == b[i])
			continue;
		else
			break;
	if (a[i] >= b[i])
		return 1;
	else
		return 0;
}

void
add_numbers(void)
{
	double a, b;
	if (isrational(stack[tos - 1]) && isrational(stack[tos - 2])) {
		qadd();
		return;
	}
	save();
	p2 = pop();
	p1 = pop();
	if (isdouble(p1))
		a = p1->u.d;
	else
		a = convert_rational_to_double(p1);
	if (isdouble(p2))
		b = p2->u.d;
	else
		b = convert_rational_to_double(p2);
	push_double(a + b);
	restore();
}

void
subtract_numbers(void)
{
	double a, b;
	if (isrational(stack[tos - 1]) && isrational(stack[tos - 2])) {
		qsub();
		return;
	}
	save();
	p2 = pop();
	p1 = pop();
	if (isdouble(p1))
		a = p1->u.d;
	else
		a = convert_rational_to_double(p1);
	if (isdouble(p2))
		b = p2->u.d;
	else
		b = convert_rational_to_double(p2);
	push_double(a - b);
	restore();
}

void
multiply_numbers(void)
{
	double a, b;
	if (isrational(stack[tos - 1]) && isrational(stack[tos - 2])) {
		qmul();
		return;
	}
	save();
	p2 = pop();
	p1 = pop();
	if (isdouble(p1))
		a = p1->u.d;
	else
		a = convert_rational_to_double(p1);
	if (isdouble(p2))
		b = p2->u.d;
	else
		b = convert_rational_to_double(p2);
	push_double(a * b);
	restore();
}

void
divide_numbers(void)
{
	double a, b;
	if (isrational(stack[tos - 1]) && isrational(stack[tos - 2])) {
		qdiv();
		return;
	}
	save();
	p2 = pop();
	p1 = pop();
	if (iszero(p2))
		stop("divide by zero");
	if (isdouble(p1))
		a = p1->u.d;
	else
		a = convert_rational_to_double(p1);
	if (isdouble(p2))
		b = p2->u.d;
	else
		b = convert_rational_to_double(p2);
	push_double(a / b);
	restore();
}

void
invert_number(void)
{
	unsigned int *a, *b;
	save();
	p1 = pop();
	if (iszero(p1))
		stop("divide by zero");
	if (isdouble(p1)) {
		push_double(1 / p1->u.d);
		restore();
		return;
	}
	a = mcopy(p1->u.q.a);
	b = mcopy(p1->u.q.b);
	MSIGN(b) = MSIGN(a);
	MSIGN(a) = 1;
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = b;
	p1->u.q.b = a;
	push(p1);
	restore();
}

int
compare_rationals(U *a, U *b)
{
	int t;
	unsigned int *ab, *ba;
	ab = mmul(a->u.q.a, b->u.q.b);
	ba = mmul(a->u.q.b, b->u.q.a);
	t = mcmp(ab, ba);
	mfree(ab);
	mfree(ba);
	return t;
}

int
compare_numbers(U *a, U *b)
{
	double x, y;
	if (isrational(a) && isrational(b))
		return compare_rationals(a, b);
	if (isdouble(a))
		x = a->u.d;
	else
		x = convert_rational_to_double(a);
	if (isdouble(b))
		y = b->u.d;
	else
		y = convert_rational_to_double(b);
	if (x < y)
		return -1;
	if (x > y)
		return 1;
	return 0;
}

void
negate_number(void)
{
	save();
	p1 = pop();
	if (iszero(p1)) {
		push(p1);
		restore();
		return;
	}
	switch (p1->k) {
	case NUM:
		p2 = alloc();
		p2->k = NUM;
		p2->u.q.a = mcopy(p1->u.q.a);
		p2->u.q.b = mcopy(p1->u.q.b);
		MSIGN(p2->u.q.a) *= -1;
		push(p2);
		break;
	case DOUBLE:
		push_double(-p1->u.d);
		break;
	default:
		stop("bug caught in mp_negate_number");
		break;
	}
	restore();
}

void
bignum_truncate(void)
{
	unsigned int *a;
	save();
	p1 = pop();
	a = mdiv(p1->u.q.a, p1->u.q.b);
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = a;
	p1->u.q.b = mint(1);
	push(p1);
	restore();
}

void
mp_numerator(void)
{
	save();
	p1 = pop();
	if (p1->k != NUM) {
		push(one);
		restore();
		return;
	}
	p2 = alloc();
	p2->k = NUM;
	p2->u.q.a = mcopy(p1->u.q.a);
	p2->u.q.b = mint(1);
	push(p2);
	restore();
}

void
mp_denominator(void)
{
	save();
	p1 = pop();
	if (p1->k != NUM) {
		push(one);
		restore();
		return;
	}
	p2 = alloc();
	p2->k = NUM;
	p2->u.q.a = mcopy(p1->u.q.b);
	p2->u.q.b = mint(1);
	push(p2);
	restore();
}

void
bignum_power_number(int expo)
{
	unsigned int *a, *b, *t;
	save();
	p1 = pop();
	a = mpow(p1->u.q.a, abs(expo));
	b = mpow(p1->u.q.b, abs(expo));
	if (expo < 0) {
		t = a;
		a = b;
		b = t;
		MSIGN(a) = MSIGN(b);
		MSIGN(b) = 1;
	}
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = a;
	p1->u.q.b = b;
	push(p1);
	restore();
}

double
convert_bignum_to_double(unsigned int *p)
{
	int i;
	double d = 0.0;
	for (i = MLENGTH(p) - 1; i >= 0; i--)
		d = 4294967296.0 * d + p[i];
	if (MSIGN(p) == -1)
		d = -d;
	return d;
}

double
convert_rational_to_double(U *p)
{
	int i, n, na, nb;
	double a = 0.0, b = 0.0;
	na = MLENGTH(p->u.q.a);
	nb = MLENGTH(p->u.q.b);
	if (na < nb)
		n = na;
	else
		n = nb;
	for (i = 0; i < n; i++) {
		a = a / 4294967296.0 + p->u.q.a[i];
		b = b / 4294967296.0 + p->u.q.b[i];
	}
	if (na > nb)
		for (i = nb; i < na; i++) {
			a = a / 4294967296.0 + p->u.q.a[i];
			b = b / 4294967296.0;
		}
	if (na < nb)
		for (i = na; i < nb; i++) {
			a = a / 4294967296.0;
			b = b / 4294967296.0 + p->u.q.b[i];
		}
	if (MSIGN(p->u.q.a) == -1)
		a = -a;
	return a / b;
}

void
push_integer(int n)
{
	save();
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = mint(n);
	p1->u.q.b = mint(1);
	push(p1);
	restore();
}

void
push_double(double d)
{
	save();
	p1 = alloc();
	p1->k = DOUBLE;
	p1->u.d = d;
	push(p1);
	restore();
}

void
push_rational(int a, int b)
{
	save();
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = mint(a);
	p1->u.q.b = mint(b);
	/* FIXME -- normalize */
	push(p1);
	restore();
}

int
pop_integer(void)
{
	int n;
	save();
	p1 = pop();
	switch (p1->k) {
	case NUM:
		if (isinteger(p1) && MLENGTH(p1->u.q.a) == 1) {
			n = p1->u.q.a[0];
			if (n & 0x80000000)
				n = 0x80000000;
			else
				n *= MSIGN(p1->u.q.a);
		} else
			n = 0x80000000;
		break;
	case DOUBLE:
		n = (int) p1->u.d;
		if ((double) n != p1->u.d)
			n = 0x80000000;
		break;
	default:
		n = 0x80000000;
		break;
	}
	restore();
	return n;
}

void
print_double(U *p, int flag)
{
	static char buf[80];
	sprintf(buf, "%g", p->u.d);
	if (flag == 1 && *buf == '-')
		print_str(buf + 1);
	else
		print_str(buf);
}

void
bignum_scan_integer(char *s)
{
	unsigned int *a;
	char sign;
	save();
	sign = *s;
	if (sign == '+' || sign == '-')
		s++;
	a = mscan(s);
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = a;
	p1->u.q.b = mint(1);
	push(p1);
	if (sign == '-')
		negate();
	restore();
}

void
bignum_scan_float(char *s)
{
	push_double(atof(s));
}

// print as unsigned

void
print_number(U *p)
{
	char *s;
	static char buf[100];
	switch (p->k) {
	case NUM:
		s = mstr(p->u.q.a);
		if (*s == '+' || *s == '-')
			s++;
		print_str(s);
		if (isfraction(p)) {
			print_str("/");
			s = mstr(p->u.q.b);
			print_str(s);
		}
		break;
	case DOUBLE:
		sprintf(buf, "%g", p->u.d);
		if (*buf == '+' || *buf == '-')
			print_str(buf + 1);
		else
			print_str(buf);
		break;
	default:
		break;
	}
}

void
gcd_numbers(void)
{
	save();
	p2 = pop();
	p1 = pop();
//	if (!isinteger(p1) || !isinteger(p2))
//		stop("integer args expected for gcd");
	p3 = alloc();
	p3->k = NUM;
	p3->u.q.a = mgcd(p1->u.q.a, p2->u.q.a);
	p3->u.q.b = mgcd(p1->u.q.b, p2->u.q.b);
	MSIGN(p3->u.q.a) = 1;
	push(p3);
	restore();
}

double
pop_double(void)
{
	double d;
	save();
	p1 = pop();
	switch (p1->k) {
	case NUM:
		d = convert_rational_to_double(p1);
		break;
	case DOUBLE:
		d = p1->u.d;
		break;
	default:
		d = 0.0;
		break;
	}
	restore();
	return d;
}

void
bignum_float(void)
{
	double d;
	d = convert_rational_to_double(pop());
	push_double(d);
}

void
bignum_factorial(int n)
{
	save();
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = bignum_factorial_nib(n);
	p1->u.q.b = mint(1);
	push(p1);
	restore();
}

unsigned int *
bignum_factorial_nib(int n)
{
	int i;
	unsigned int *a, *b, *t;
	if (n == 0 || n == 1) {
		a = mint(1);
		return a;
	}
	a = mint(2);
	b = mint(0);
	for (i = 3; i <= n; i++) {
		b[0] = (unsigned int) i;
		t = mmul(a, b);
		mfree(a);
		a = t;
	}
	mfree(b);
	return a;
}

unsigned int mask[32] = {
	0x00000001,
	0x00000002,
	0x00000004,
	0x00000008,
	0x00000010,
	0x00000020,
	0x00000040,
	0x00000080,
	0x00000100,
	0x00000200,
	0x00000400,
	0x00000800,
	0x00001000,
	0x00002000,
	0x00004000,
	0x00008000,
	0x00010000,
	0x00020000,
	0x00040000,
	0x00080000,
	0x00100000,
	0x00200000,
	0x00400000,
	0x00800000,
	0x01000000,
	0x02000000,
	0x04000000,
	0x08000000,
	0x10000000,
	0x20000000,
	0x40000000,
	0x80000000,
};

void
mp_set_bit(unsigned int *x, unsigned int k)
{
	x[k / 32] |= mask[k % 32];
}

void
mp_clr_bit(unsigned int *x, unsigned int k)
{
	x[k / 32] &= ~mask[k % 32];
}

void
mshiftright(unsigned int *a)
{
	int c, i, n;
	n = MLENGTH(a);
	c = 0;
	for (i = n - 1; i >= 0; i--)
		if (a[i] & 1) {
			a[i] = (a[i] >> 1) | c;
			c = 0x80000000;
		} else {
			a[i] = (a[i] >> 1) | c;
			c = 0;
		}
	if (n > 1 && a[n - 1] == 0)
		MLENGTH(a) = n - 1;
}

//	Binomial coefficient
//
//	Input:		tos-2		n
//
//			tos-1		k
//
//	Output:		Binomial coefficient on stack
//
//	binomial(n, k) = n! / k! / (n - k)!
//
//	The binomial coefficient vanishes for k < 0 or k > n. (A=B, p. 19)

void
eval_binomial(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	binomial();
}

void
binomial(void)
{
	save();
	binomial_nib();
	restore();
}

#undef N
#undef K

#define N p1
#define K p2

void
binomial_nib(void)
{
	K = pop();
	N = pop();
	if (binomial_check_args() == 0) {
		push(zero);
		return;
	}
	push(N);
	factorial();
	push(K);
	factorial();
	divide();
	push(N);
	push(K);
	subtract();
	factorial();
	divide();
}

int
binomial_check_args(void)
{
	if (isnum(N) && lessp(N, zero))
		return 0;
	else if (isnum(K) && lessp(K, zero))
		return 0;
	else if (isnum(N) && isnum(K) && lessp(N, K))
		return 0;
	else
		return 1;
}

void
eval_ceiling(void)
{
	push(cadr(p1));
	eval();
	ceiling();
}

void
ceiling(void)
{
	save();
	yyceiling();
	restore();
}

void
yyceiling(void)
{
	double d;
	p1 = pop();
	if (!isnum(p1)) {
		push_symbol(CEILING);
		push(p1);
		list(2);
		return;
	}
	if (isdouble(p1)) {
		d = ceil(p1->u.d);
		push_double(d);
		return;
	}
	if (isinteger(p1)) {
		push(p1);
		return;
	}
	p3 = alloc();
	p3->k = NUM;
	p3->u.q.a = mdiv(p1->u.q.a, p1->u.q.b);
	p3->u.q.b = mint(1);
	push(p3);
	if (isnegativenumber(p1))
		;
	else {
		push_integer(1);
		add();
	}
}

// For example, the number of five card hands is choose(52,5)
//
//                          n!
//      choose(n,k) = -------------
//                     k! (n - k)!

void
eval_choose(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	choose();
}

// Result vanishes for k < 0 or k > n. (A=B, p. 19)

#undef N
#undef K

#define N p1
#define K p2

void
choose(void)
{
	save();
	K = pop();
	N = pop();
	if (choose_check_args() == 0) {
		push_integer(0);
		restore();
		return;
	}
	push(N);
	factorial();
	push(K);
	factorial();
	divide();
	push(N);
	push(K);
	subtract();
	factorial();
	divide();
	restore();
}

int
choose_check_args(void)
{
	if (isnum(N) && lessp(N, zero))
		return 0;
	else if (isnum(K) && lessp(K, zero))
		return 0;
	else if (isnum(N) && isnum(K) && lessp(N, K))
		return 0;
	else
		return 1;
}

// Change circular functions to exponentials

void
eval_circexp(void)
{
	push(cadr(p1));
	eval();
	circexp();
	// normalize
	eval();
}

void
circexp(void)
{
	int i, h;
	save();
	p1 = pop();
	if (car(p1) == symbol(COS)) {
		push(cadr(p1));
		expcos();
		restore();
		return;
	}
	if (car(p1) == symbol(SIN)) {
		push(cadr(p1));
		expsin();
		restore();
		return;
	}
	if (car(p1) == symbol(TAN)) {
		p1 = cadr(p1);
		push(imaginaryunit);
		push(p1);
		multiply();
		exponential();
		p2 = pop();
		push(imaginaryunit);
		push(p1);
		multiply();
		negate();
		exponential();
		p3 = pop();
		push(p3);
		push(p2);
		subtract();
		push(imaginaryunit);
		multiply();
		push(p2);
		push(p3);
		add();
		divide();
		restore();
		return;
	}
	if (car(p1) == symbol(COSH)) {
		p1 = cadr(p1);
		push(p1);
		exponential();
		push(p1);
		negate();
		exponential();
		add();
		push_rational(1, 2);
		multiply();
		restore();
		return;
	}
	if (car(p1) == symbol(SINH)) {
		p1 = cadr(p1);
		push(p1);
		exponential();
		push(p1);
		negate();
		exponential();
		subtract();
		push_rational(1, 2);
		multiply();
		restore();
		return;
	}
	if (car(p1) == symbol(TANH)) {
		p1 = cadr(p1);
		push(p1);
		push_integer(2);
		multiply();
		exponential();
		p1 = pop();
		push(p1);
		push_integer(1);
		subtract();
		push(p1);
		push_integer(1);
		add();
		divide();
		restore();
		return;
	}
	if (iscons(p1)) {
		h = tos;
		while (iscons(p1)) {
			push(car(p1));
			circexp();
			p1 = cdr(p1);
		}
		list(tos - h);
		restore();
		return;
	}
	if (p1->k == TENSOR) {
		push(p1);
		copy_tensor();
		p1 = pop();
		for (i = 0; i < p1->u.tensor->nelem; i++) {
			push(p1->u.tensor->elem[i]);
			circexp();
			p1->u.tensor->elem[i] = pop();
		}
		push(p1);
		restore();
		return;
	}
	push(p1);
	restore();
}

char *script =
"e=exp(1)\n"
"i=sqrt(-1)\n"
"autoexpand=1\n"
"bake=1\n"
"trange=(-pi,pi)\n"
"xrange=(-10,10)\n"
"yrange=(-10,10)\n"
"last=0\n"
"trace=0\n"
"tty=0\n"
"cross(u,v)=(u[2]*v[3]-u[3]*v[2],u[3]*v[1]-u[1]*v[3],u[1]*v[2]-u[2]*v[1])\n"
"curl(v)=(d(v[3],y)-d(v[2],z),d(v[1],z)-d(v[3],x),d(v[2],x)-d(v[1],y))\n"
"div(v)=d(v[1],x)+d(v[2],y)+d(v[3],z)\n"
"ln(x)=log(x)\n";

void
clear(void)
{
	int i;
	// reset bindings
	for (i = MARK2 + 1; i < NSYM; i++) {
		binding[i] = symtab + i;
		arglist[i] = symbol(NIL);
	}
	// free user-defined symbol names
	for (i = MARK3 + 1; i < NSYM; i++) {
		if (symtab[i].u.printname == NULL)
			break;
		free(symtab[i].u.printname);
		symtab[i].u.printname = NULL;
	}
	run(script);
	gc();
	clear_display();
}

/* Convert complex z to clock form

	Input:		push	z

	Output:		Result on stack

	clock(z) = mag(z) * (-1) ^ (arg(z) / pi)

	For example, clock(exp(i pi/3)) gives the result (-1)^(1/3)
*/

void
eval_clock(void)
{
	push(cadr(p1));
	eval();
	clockform();
}

void
clockform(void)
{
	save();
#if 1
	p1 = pop();
	push(p1);
	mag();
	push_integer(-1);
	push(p1);
	arg();
	push_symbol(PI);
	divide();
	power();
	multiply();
#else
	p1 = pop();
	push(p1);
	mag();
	push_symbol(E);
	push(p1);
	arg();
	push(imaginaryunit);
	multiply();
	power();
	multiply();
#endif
	restore();
}

// get the coefficient of x^n in polynomial p(x)

#undef P
#undef X
#undef N

#define P p1
#define X p2
#define N p3

void
eval_coeff(void)
{
	push(cadr(p1));			// 1st arg, p
	eval();
	push(caddr(p1));		// 2nd arg, x
	eval();
	push(cadddr(p1));		// 3rd arg, n
	eval();
	N = pop();
	X = pop();
	P = pop();
	if (N == symbol(NIL)) {		// only 2 args?
		N = X;
		X = symbol(SYMBOL_X);
	}
	push(P);			// divide p by x^n
	push(X);
	push(N);
	power();
	divide();
	push(X);			// keep the constant part
	filter();
}

//-----------------------------------------------------------------------------
//
//	Put polynomial coefficients on the stack
//
//	Input:		tos-2		p(x)
//
//			tos-1		x
//
//	Output:		Returns number of coefficients on stack
//
//			tos-n		Coefficient of x^0
//
//			tos-1		Coefficient of x^(n-1)
//
//-----------------------------------------------------------------------------

int
coeff(void)
{
	int h, n;
	save();
	p2 = pop();
	p1 = pop();
	h = tos;
	for (;;) {
		push(p1);
		push(p2);
		push(zero);
		subst();
		eval();
		p3 = pop();
		push(p3);
		push(p1);
		push(p3);
		subtract();
		p1 = pop();
		if (equal(p1, zero)) {
			n = tos - h;
			restore();
			return n;
		}
		push(p1);
		push(p2);
		divide();
		p1 = pop();
	}
}

// Cofactor of a matrix component.

void
eval_cofactor(void)
{
	int i, j, n;
	push(cadr(p1));
	eval();
	p2 = pop();
	if (istensor(p2) && p2->u.tensor->ndim == 2 && p2->u.tensor->dim[0] == p2->u.tensor->dim[1])
		;
	else
		stop("cofactor: 1st arg: square matrix expected");
	n = p2->u.tensor->dim[0];
	push(caddr(p1));
	eval();
	i = pop_integer();
	if (i < 1 || i > n)
		stop("cofactor: 2nd arg: row index expected");
	push(cadddr(p1));
	eval();
	j = pop_integer();
	if (j < 1 || j > n)
		stop("cofactor: 3rd arg: column index expected");
	cofactor(p2, n, i - 1, j - 1);
}

void
cofactor(U *p, int n, int row, int col)
{
	int i, j;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			if (i != row && j != col)
				push(p->u.tensor->elem[n * i + j]);
	determinant(n - 1);
	if ((row + col) % 2)
		negate();
}

// Condense an expression by factoring common terms.

void
eval_condense(void)
{
	push(cadr(p1));
	eval();
	Condense();
}

void
Condense(void)
{
	int tmp;
	tmp = expanding;
	save();
	yycondense();
	restore();
	expanding = tmp;
}

void
yycondense(void)
{
	expanding = 0;
	p1 = pop();
	if (car(p1) != symbol(ADD)) {
		push(p1);
		return;
	}
	// get gcd of all terms
	p3 = cdr(p1);
	push(car(p3));
	p3 = cdr(p3);
	while (iscons(p3)) {
		push(car(p3));
		gcd();
		p3 = cdr(p3);
	}
//printf("condense: this is the gcd of all the terms:\n");
//print(stdout, stack[tos - 1]);
	// divide each term by gcd
	inverse();
	p2 = pop();
	push(zero);
	p3 = cdr(p1);
	while (iscons(p3)) {
		push(p2);
		push(car(p3));
		multiply();
		add();
		p3 = cdr(p3);
	}
	// We multiplied above w/o expanding so sum factors cancelled.
	// Now we expand which which normalizes the result and, in some cases,
	// simplifies it too (see test case H).
	yyexpand();
	// multiply result by gcd
	push(p2);
	divide();
}

// Complex conjugate

void
eval_conj(void)
{
	push(cadr(p1));
	eval();
	p1 = pop();
	push(p1);
	conjugate();
}

void
conjugate(void)
{
	push(imaginaryunit);
	push(imaginaryunit);
	negate();
	subst();
	eval();
}

// Cons two things on the stack.

void
cons(void)
{
	// auto var ok, no opportunity for garbage collection after p = alloc()
	U *p;
	p = alloc();
	p->k = CONS;
	p->u.cons.cdr = pop();
	p->u.cons.car = pop();
	push(p);
}

// Contract across tensor indices

void
eval_contract(void)
{
	push(cadr(p1));
	eval();
	if (cddr(p1) == symbol(NIL)) {
		push_integer(1);
		push_integer(2);
	} else {
		push(caddr(p1));
		eval();
		push(cadddr(p1));
		eval();
	}
	contract();
}

void
contract(void)
{
	save();
	yycontract();
	restore();
}

void
yycontract(void)
{
	int h, i, j, k, l, m, n, ndim, nelem;
	int ai[MAXDIM], an[MAXDIM];
	U **a, **b;
	p3 = pop();
	p2 = pop();
	p1 = pop();
	if (!istensor(p1)) {
		if (!iszero(p1))
			stop("contract: tensor expected, 1st arg is not a tensor");
		push(zero);
		return;
	}
	push(p2);
	l = pop_integer();
	push(p3);
	m = pop_integer();
	ndim = p1->u.tensor->ndim;
	if (l < 1 || l > ndim || m < 1 || m > ndim || l == m
	|| p1->u.tensor->dim[l - 1] != p1->u.tensor->dim[m - 1])
		stop("contract: index out of range");
	l--;
	m--;
	n = p1->u.tensor->dim[l];
	// nelem is the number of elements in "b"
	nelem = 1;
	for (i = 0; i < ndim; i++)
		if (i != l && i != m)
			nelem *= p1->u.tensor->dim[i];
	p2 = alloc_tensor(nelem);
	p2->u.tensor->ndim = ndim - 2;
	j = 0;
	for (i = 0; i < ndim; i++)
		if (i != l && i != m)
			p2->u.tensor->dim[j++] = p1->u.tensor->dim[i];
	a = p1->u.tensor->elem;
	b = p2->u.tensor->elem;
	for (i = 0; i < ndim; i++) {
		ai[i] = 0;
		an[i] = p1->u.tensor->dim[i];
	}
	for (i = 0; i < nelem; i++) {
		push(zero);
		for (j = 0; j < n; j++) {
			ai[l] = j;
			ai[m] = j;
			h = 0;
			for (k = 0; k < ndim; k++)
				h = (h * an[k]) + ai[k];
			push(a[h]);
			add();
		}
		b[i] = pop();
		for (j = ndim - 1; j >= 0; j--) {
			if (j == l || j == m)
				continue;
			if (++ai[j] < an[j])
				break;
			ai[j] = 0;
		}
	}
	if (nelem == 1)
		push(b[0]);
	else
		push(p2);
}

void
eval_cos(void)
{
	push(cadr(p1));
	eval();
	cosine();
}

void
cosine(void)
{
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD))
		cosine_of_angle_sum();
	else
		cosine_of_angle();
	restore();
}

// Use angle sum formula for special angles.

#undef A
#undef B

#define A p3
#define B p4

void
cosine_of_angle_sum(void)
{
	p2 = cdr(p1);
	while (iscons(p2)) {
		B = car(p2);
		if (isnpi(B)) {
			push(p1);
			push(B);
			subtract();
			A = pop();
			push(A);
			cosine();
			push(B);
			cosine();
			multiply();
			push(A);
			sine();
			push(B);
			sine();
			multiply();
			subtract();
			return;
		}
		p2 = cdr(p2);
	}
	cosine_of_angle();
}

void
cosine_of_angle(void)
{
	int n;
	double d;
	if (car(p1) == symbol(ARCCOS)) {
		push(cadr(p1));
		return;
	}
	if (isdouble(p1)) {
		d = cos(p1->u.d);
		if (fabs(d) < 1e-10)
			d = 0.0;
		push_double(d);
		return;
	}
	// cosine function is symmetric, cos(-x) = cos(x)
	if (isnegative(p1)) {
		push(p1);
		negate();
		p1 = pop();
	}
	// cos(arctan(x)) = 1 / sqrt(1 + x^2)
	// see p. 173 of the CRC Handbook of Mathematical Sciences
	if (car(p1) == symbol(ARCTAN)) {
		push_integer(1);
		push(cadr(p1));
		push_integer(2);
		power();
		add();
		push_rational(-1, 2);
		power();
		return;
	}
	// multiply by 180/pi
	push(p1);
	push_integer(180);
	multiply();
	push_symbol(PI);
	divide();
	n = pop_integer();
	if (n < 0) {
		push(symbol(COS));
		push(p1);
		list(2);
		return;
	}
	switch (n % 360) {
	case 90:
	case 270:
		push_integer(0);
		break;
	case 60:
	case 300:
		push_rational(1, 2);
		break;
	case 120:
	case 240:
		push_rational(-1, 2);
		break;
	case 45:
	case 315:
		push_rational(1, 2);
		push_integer(2);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 135:
	case 225:
		push_rational(-1, 2);
		push_integer(2);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 30:
	case 330:
		push_rational(1, 2);
		push_integer(3);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 150:
	case 210:
		push_rational(-1, 2);
		push_integer(3);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 0:
		push_integer(1);
		break;
	case 180:
		push_integer(-1);
		break;
	default:
		push(symbol(COS));
		push(p1);
		list(2);
		break;
	}
}

//	          exp(x) + exp(-x)
//	cosh(x) = ----------------
//	                 2

void
eval_cosh(void)
{
	push(cadr(p1));
	eval();
	ycosh();
}

void
ycosh(void)
{
	save();
	yycosh();
	restore();
}

void
yycosh(void)
{
	double d;
	p1 = pop();
	if (car(p1) == symbol(ARCCOSH)) {
		push(cadr(p1));
		return;
	}
	if (isdouble(p1)) {
		d = cosh(p1->u.d);
		if (fabs(d) < 1e-10)
			d = 0.0;
		push_double(d);
		return;
	}
	if (iszero(p1)) {
		push(one);
		return;
	}
	push_symbol(COSH);
	push(p1);
	list(2);
}

void
eval_decomp(void)
{
	int h = tos;
	push(symbol(NIL));
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	p1 = pop();
	if (p1 == symbol(NIL))
		guess();
	else
		push(p1);
	decomp_nib();
	list(tos - h);
}

// returns constant expresions on the stack

void
decomp_nib(void)
{
	save();
	p2 = pop();
	p1 = pop();
	// is the entire expression constant?
	if (find(p1, p2) == 0) {
		push(p1);
		//push(p1);	// may need later for pushing both +a, -a
		//negate();
		restore();
		return;
	}
	// sum?
	if (isadd(p1)) {
		decomp_sum();
		restore();
		return;
	}
	// product?
	if (car(p1) == symbol(MULTIPLY)) {
		decomp_product();
		restore();
		return;
	}
	// naive decomp if not sum or product
	p3 = cdr(p1);
	while (iscons(p3)) {
		push(car(p3));
		push(p2);
		decomp_nib();
		p3 = cdr(p3);
	}
	restore();
}

void
decomp_sum(void)
{
	int h;
	// decomp terms involving x
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2)) {
			push(car(p3));
			push(p2);
			decomp_nib();
		}
		p3 = cdr(p3);
	}
	// add together all constant terms
	h = tos;
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2) == 0)
			push(car(p3));
		p3 = cdr(p3);
	}
	if (tos - h) {
		add_all(tos - h);
		p3 = pop();
		push(p3);
		push(p3);
		negate();	// need both +a, -a for some integrals
	}
}

void
decomp_product(void)
{
	int h;
	// decomp factors involving x
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2)) {
			push(car(p3));
			push(p2);
			decomp_nib();
		}
		p3 = cdr(p3);
	}
	// multiply together all constant factors
	h = tos;
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2) == 0)
			push(car(p3));
		p3 = cdr(p3);
	}
	if (tos - h) {
		multiply_all(tos - h);
		//p3 = pop();	// may need later for pushing both +a, -a
		//push(p3);
		//push(p3);
		//negate();
	}
}

// Store a function definition
//
// Example:
//
//      f(x,y)=x^y
//
// For this definition, p1 points to the following structure.
//
//     p1
//      |
//   ___v__    ______                        ______
//  |CONS  |->|CONS  |--------------------->|CONS  |
//  |______|  |______|                      |______|
//      |         |                             |
//   ___v__    ___v__    ______    ______    ___v__    ______    ______
//  |SETQ  |  |CONS  |->|CONS  |->|CONS  |  |CONS  |->|CONS  |->|CONS  |
//  |______|  |______|  |______|  |______|  |______|  |______|  |______|
//                |         |         |         |         |         |
//             ___v__    ___v__    ___v__    ___v__    ___v__    ___v__
//            |SYM f |  |SYM x |  |SYM y |  |POWER |  |SYM x |  |SYM y |
//            |______|  |______|  |______|  |______|  |______|  |______|
//
// We have
//
//	caadr(p1) points to f
//	cdadr(p1) points to the list (x y)
//	caddr(p1) points to (power x y)

#undef F
#undef A
#undef B

#define F p3 // F points to the function name
#define A p4 // A points to the argument list
#define B p5 // B points to the function body

void
define_user_function(void)
{
	F = caadr(p1);
	A = cdadr(p1);
	B = caddr(p1);
	if (!issymbol(F))
		stop("function name?");
	// evaluate function body (maybe)
	if (car(B) == symbol(EVAL)) {
		push(cadr(B));
		eval();
		B = pop();
	}
	set_binding_and_arglist(F, B, A);
	// return value is nil
	push(symbol(NIL));
}

// definite integral

#undef F
#undef X
#undef A
#undef B

#define F p2
#define X p3
#define A p4
#define B p5

void
eval_defint(void)
{
	push(cadr(p1));
	eval();
	F = pop();
	p1 = cddr(p1);
	while (iscons(p1)) {
		push(car(p1));
		p1 = cdr(p1);
		eval();
		X = pop();
		push(car(p1));
		p1 = cdr(p1);
		eval();
		A = pop();
		push(car(p1));
		p1 = cdr(p1);
		eval();
		B = pop();
		push(F);
		push(X);
		integral();
		F = pop();
		push(F);
		push(X);
		push(B);
		subst();
		eval();
		push(F);
		push(X);
		push(A);
		subst();
		eval();
		subtract();
		F = pop();
	}
	push(F);
}

void
eval_degree(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	p1 = pop();
	if (p1 == symbol(NIL))
		guess();
	else
		push(p1);
	degree();
}

//-----------------------------------------------------------------------------
//
//	Find the degree of a polynomial
//
//	Input:		tos-2		p(x)
//
//			tos-1		x
//
//	Output:		Result on stack
//
//	Note: Finds the largest numerical power of x. Does not check for
//	weirdness in p(x).
//
//-----------------------------------------------------------------------------

#undef POLY
#undef X
#undef YDEGREE

#define POLY p1
#define X p2
#define YDEGREE p3

void
degree(void)
{
	save();
	X = pop();
	POLY = pop();
	YDEGREE = zero;
	yydegree(POLY);
	push(YDEGREE);
	restore();
}

void
yydegree(U *p)
{
	if (equal(p, X)) {
		if (iszero(YDEGREE))
			YDEGREE = one;
	} else if (car(p) == symbol(POWER)) {
		if (equal(cadr(p), X) && isnum(caddr(p)) && lessp(YDEGREE, caddr(p)))
			YDEGREE = caddr(p);
	} else if (iscons(p)) {
		p = cdr(p);
		while (iscons(p)) {
			yydegree(car(p));
			p = cdr(p);
		}
	}
}

void
eval_denominator(void)
{
	push(cadr(p1));
	eval();
	denominator();
}

void
denominator(void)
{
	int h;
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD)) {
		push(p1);
		rationalize();
		p1 = pop();
	}
	if (car(p1) == symbol(MULTIPLY)) {
		h = tos;
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			denominator();
			p1 = cdr(p1);
		}
		multiply_all(tos - h);
	} else if (isrational(p1)) {
		push(p1);
		mp_denominator();
	} else if (car(p1) == symbol(POWER) && isnegativeterm(caddr(p1))) {
		push(p1);
		reciprocate();
	} else
		push(one);
	restore();
}

#undef F
#undef X
#undef N

#define F p3
#define X p4
#define N p5

void
eval_derivative(void)
{
	int i, n;
	// evaluate 1st arg to get function F
	p1 = cdr(p1);
	push(car(p1));
	eval();
	// evaluate 2nd arg and then...
	// example	result of 2nd arg	what to do
	//
	// d(f)		nil			guess X, N = nil
	// d(f,2)	2			guess X, N = 2
	// d(f,x)	x			X = x, N = nil
	// d(f,x,2)	x			X = x, N = 2
	// d(f,x,y)	x			X = x, N = y
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL)) {
		guess();
		push(symbol(NIL));
	} else if (isnum(p2)) {
		guess();
		push(p2);
	} else {
		push(p2);
		p1 = cdr(p1);
		push(car(p1));
		eval();
	}
	N = pop();
	X = pop();
	F = pop();
	while (1) {
		// N might be a symbol instead of a number
		if (isnum(N)) {
			push(N);
			n = pop_integer();
			if (n == (int) 0x80000000)
				stop("nth derivative: check n");
		} else
			n = 1;
		push(F);
		if (n >= 0) {
			for (i = 0; i < n; i++) {
				push(X);
				derivative();
			}
		} else {
			n = -n;
			for (i = 0; i < n; i++) {
				push(X);
				integral();
			}
		}
		F = pop();
		// if N is nil then arglist is exhausted
		if (N == symbol(NIL))
			break;
		// otherwise...
		// N		arg1		what to do
		//
		// number	nil		break
		// number	number		N = arg1, continue
		// number	symbol		X = arg1, N = arg2, continue
		//
		// symbol	nil		X = N, N = nil, continue
		// symbol	number		X = N, N = arg1, continue
		// symbol	symbol		X = N, N = arg1, continue
		if (isnum(N)) {
			p1 = cdr(p1);
			push(car(p1));
			eval();
			N = pop();
			if (N == symbol(NIL))
				break;		// arglist exhausted
			if (isnum(N))
				;		// N = arg1
			else {
				X = N;		// X = arg1
				p1 = cdr(p1);
				push(car(p1));
				eval();
				N = pop();	// N = arg2
			}
		} else {
			X = N;			// X = N
			p1 = cdr(p1);
			push(car(p1));
			eval();
			N = pop();		// N = arg1
		}
	}
	push(F); // final result
}

void
derivative(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (isnum(p2))
		stop("undefined function");
	if (istensor(p1))
		if (istensor(p2))
			d_tensor_tensor();
		else
			d_tensor_scalar();
	else
		if (istensor(p2))
			d_scalar_tensor();
		else
			d_scalar_scalar();
	restore();
}

void
d_scalar_scalar(void)
{
	if (issymbol(p2))
		d_scalar_scalar_1();
	else {
		// Example: d(sin(cos(x)),cos(x))
		// Replace cos(x) <- X, find derivative, then do X <- cos(x)
		push(p1);		// sin(cos(x))
		push(p2);		// cos(x)
		push_symbol(SPECX);	// X
		subst();		// sin(cos(x)) -> sin(X)
		push_symbol(SPECX);	// X
		derivative();
		push_symbol(SPECX);	// X
		push(p2);		// cos(x)
		subst();		// cos(X) -> cos(cos(x))
	}
}

void
d_scalar_scalar_1(void)
{
	// d(x,x)?
	if (equal(p1, p2)) {
		push(one);
		return;
	}
	// d(a,x)?
	if (!iscons(p1)) {
		push(zero);
		return;
	}
	if (isadd(p1)) {
		dsum();
		return;
	}
	if (car(p1) == symbol(MULTIPLY)) {
		dproduct();
		return;
	}
	if (car(p1) == symbol(POWER)) {
		dpower();
		return;
	}
	if (car(p1) == symbol(DERIVATIVE)) {
		dd();
		return;
	}
	if (car(p1) == symbol(LOG)) {
		dlog();
		return;
	}
	if (car(p1) == symbol(SIN)) {
		dsin();
		return;
	}
	if (car(p1) == symbol(COS)) {
		dcos();
		return;
	}
	if (car(p1) == symbol(TAN)) {
		dtan();
		return;
	}
	if (car(p1) == symbol(ARCSIN)) {
		darcsin();
		return;
	}
	if (car(p1) == symbol(ARCCOS)) {
		darccos();
		return;
	}
	if (car(p1) == symbol(ARCTAN)) {
		darctan();
		return;
	}
	if (car(p1) == symbol(SINH)) {
		dsinh();
		return;
	}
	if (car(p1) == symbol(COSH)) {
		dcosh();
		return;
	}
	if (car(p1) == symbol(TANH)) {
		dtanh();
		return;
	}
	if (car(p1) == symbol(ARCSINH)) {
		darcsinh();
		return;
	}
	if (car(p1) == symbol(ARCCOSH)) {
		darccosh();
		return;
	}
	if (car(p1) == symbol(ARCTANH)) {
		darctanh();
		return;
	}
	if (car(p1) == symbol(ABS)) {
		dabs();
		return;
	}
	if (car(p1) == symbol(HERMITE)) {
		dhermite();
		return;
	}
	if (car(p1) == symbol(ERF)) {
		derf();
		return;
	}
	if (car(p1) == symbol(ERFC)) {
		derfc();
		return;
	}
	if (car(p1) == symbol(BESSELJ)) {
		if (iszero(caddr(p1)))
			dbesselj0();
		else
			dbesseljn();
		return;
	}
	if (car(p1) == symbol(BESSELY)) {
		if (iszero(caddr(p1)))
			dbessely0();
		else
			dbesselyn();
		return;
	}
	if (car(p1) == symbol(INTEGRAL) && caddr(p1) == p2) {
		derivative_of_integral();
		return;
	}
	dfunction();
}

void
dsum(void)
{
	int h = tos;
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		push(p2);
		derivative();
		p1 = cdr(p1);
	}
	add_all(tos - h);
}

void
dproduct(void)
{
	int i, j, n;
	n = length(p1) - 1;
	for (i = 0; i < n; i++) {
		p3 = cdr(p1);
		for (j = 0; j < n; j++) {
			push(car(p3));
			if (i == j) {
				push(p2);
				derivative();
			}
			p3 = cdr(p3);
		}
		multiply_all(n);
	}
	add_all(n);
}

//-----------------------------------------------------------------------------
//
//	     v
//	y = u
//
//	log y = v log u
//
//	1 dy   v du           dv
//	- -- = - -- + (log u) --
//	y dx   u dx           dx
//
//	dy    v  v du           dv
//	-- = u  (- -- + (log u) --)
//	dx       u dx           dx
//
//-----------------------------------------------------------------------------

void
dpower(void)
{
	push(caddr(p1));	// v/u
	push(cadr(p1));
	divide();
	push(cadr(p1));		// du/dx
	push(p2);
	derivative();
	multiply();
	push(cadr(p1));		// log u
	logarithm();
	push(caddr(p1));	// dv/dx
	push(p2);
	derivative();
	multiply();
	add();
	push(p1);		// u^v
	multiply();
}

void
dlog(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	divide();
}

//	derivative of derivative
//
//	example: d(d(f(x,y),y),x)
//
//	p1 = d(f(x,y),y)
//
//	p2 = x
//
//	cadr(p1) = f(x,y)
//
//	caddr(p1) = y

void
dd(void)
{
	// d(f(x,y),x)
	push(cadr(p1));
	push(p2);
	derivative();
	p3 = pop();
	if (car(p3) == symbol(DERIVATIVE)) {
		// sort dx terms
		push_symbol(DERIVATIVE);
		push_symbol(DERIVATIVE);
		push(cadr(p3));
		if (lessp(caddr(p3), caddr(p1))) {
			push(caddr(p3));
			list(3);
			push(caddr(p1));
		} else {
			push(caddr(p1));
			list(3);
			push(caddr(p3));
		}
		list(3);
	} else {
		push(p3);
		push(caddr(p1));
		derivative();
	}
}

// derivative of a generic function

void
dfunction(void)
{
	p3 = cdr(p1);	// p3 is the argument list for the function
	if (p3 == symbol(NIL) || find(p3, p2)) {
		push_symbol(DERIVATIVE);
		push(p1);
		push(p2);
		list(3);
	} else
		push(zero);
}

void
dsin(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	cosine();
	multiply();
}

void
dcos(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	sine();
	multiply();
	negate();
}

void
dtan(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	cosine();
	push_integer(-2);
	power();
	multiply();
}

void
darcsin(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push_integer(1);
	push(cadr(p1));
	push_integer(2);
	power();
	subtract();
	push_rational(-1, 2);
	power();
	multiply();
}

void
darccos(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push_integer(1);
	push(cadr(p1));
	push_integer(2);
	power();
	subtract();
	push_rational(-1, 2);
	power();
	multiply();
	negate();
}

//				Without simplify	With simplify
//
//	d(arctan(y/x),x)	-y/(x^2*(y^2/x^2+1))	-y/(x^2+y^2)
//
//	d(arctan(y/x),y)	1/(x*(y^2/x^2+1))	x/(x^2+y^2)

void
darctan(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push_integer(1);
	push(cadr(p1));
	push_integer(2);
	power();
	add();
	inverse();
	multiply();
	simplify();
}

void
dsinh(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	ycosh();
	multiply();
}

void
dcosh(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	ysinh();
	multiply();
}

void
dtanh(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	ycosh();
	push_integer(-2);
	power();
	multiply();
}

void
darcsinh(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	push_integer(2);
	power();
	push_integer(1);
	add();
	push_rational(-1, 2);
	power();
	multiply();
}

void
darccosh(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	push_integer(2);
	power();
	push_integer(-1);
	add();
	push_rational(-1, 2);
	power();
	multiply();
}

void
darctanh(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push_integer(1);
	push(cadr(p1));
	push_integer(2);
	power();
	subtract();
	inverse();
	multiply();
}

void
dabs(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	sgn();
	multiply();
}

void
dhermite(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push_integer(2);
	push(caddr(p1));
	multiply();
	multiply();
	push(cadr(p1));
	push(caddr(p1));
	push_integer(-1);
	add();
	hermite();
	multiply();
}

void
derf(void)
{
	push(cadr(p1));
	push_integer(2);
	power();
	push_integer(-1);
	multiply();
	exponential();
	push_symbol(PI);
	push_rational(-1,2);
	power();
	multiply();
	push_integer(2);
	multiply();
	push(cadr(p1));
	push(p2);
	derivative();
	multiply();
}

void
derfc(void)
{
	push(cadr(p1));
	push_integer(2);
	power();
	push_integer(-1);
	multiply();
	exponential();
	push_symbol(PI);
	push_rational(-1,2);
	power();
	multiply();
	push_integer(-2);
	multiply();
	push(cadr(p1));
	push(p2);
	derivative();
	multiply();
}

void
dbesselj0(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	push_integer(1);
	besselj();
	multiply();
	push_integer(-1);
	multiply();
}

void
dbesseljn(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	push(caddr(p1));
	push_integer(-1);
	add();
	besselj();
	push(caddr(p1));
	push_integer(-1);
	multiply();
	push(cadr(p1));
	divide();
	push(cadr(p1));
	push(caddr(p1));
	besselj();
	multiply();
	add();
	multiply();
}

void
dbessely0(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	push_integer(1);
	besselj();
	multiply();
	push_integer(-1);
	multiply();
}

void
dbesselyn(void)
{
	push(cadr(p1));
	push(p2);
	derivative();
	push(cadr(p1));
	push(caddr(p1));
	push_integer(-1);
	add();
	bessely();
	push(caddr(p1));
	push_integer(-1);
	multiply();
	push(cadr(p1));
	divide();
	push(cadr(p1));
	push(caddr(p1));
	bessely();
	multiply();
	add();
	multiply();
}

void
derivative_of_integral(void)
{
	push(cadr(p1));
}

//-----------------------------------------------------------------------------
//
//	Input:		Matrix on stack
//
//	Output:		Determinant on stack
//
//	Example:
//
//	> det(((1,2),(3,4)))
//	-2
//
//	Note:
//
//	Uses Gaussian elimination for numerical matrices.
//
//-----------------------------------------------------------------------------

int
det_check_arg(void)
{
	if (!istensor(p1))
		return 0;
	else if (p1->u.tensor->ndim != 2)
		return 0;
	else if (p1->u.tensor->dim[0] != p1->u.tensor->dim[1])
		return 0;
	else
		return 1;
}

void
det(void)
{
	int i, n;
	U **a;
	save();
	p1 = pop();
	if (det_check_arg() == 0) {
		push_symbol(DET);
		push(p1);
		list(2);
		restore();
		return;
	}
	n = p1->u.tensor->nelem;
	a = p1->u.tensor->elem;
	for (i = 0; i < n; i++)
		if (!isnum(a[i]))
			break;
	if (i == n)
		yydetg();
	else {
		for (i = 0; i < p1->u.tensor->nelem; i++)
			push(p1->u.tensor->elem[i]);
		determinant(p1->u.tensor->dim[0]);
	}
	restore();
}

// determinant of n * n matrix elements on the stack

void
determinant(int n)
{
	int h, i, j, k, q, s, sign, t;
	int *a, *c, *d;
	h = tos - n * n;
	a = (int *) malloc(3 * n * sizeof (int));
	if (a == NULL)
		out_of_memory();
	c = a + n;
	d = c + n;
	for (i = 0; i < n; i++) {
		a[i] = i;
		c[i] = 0;
		d[i] = 1;
	}
	sign = 1;
	push(zero);
	for (;;) {
		if (sign == 1)
			push_integer(1);
		else
			push_integer(-1);
		for (i = 0; i < n; i++) {
			k = n * a[i] + i;
			push(stack[h + k]);
			multiply(); // FIXME -- problem here
		}
		add();
		/* next permutation (Knuth's algorithm P) */
		j = n - 1;
		s = 0;
	P4:	q = c[j] + d[j];
		if (q < 0) {
			d[j] = -d[j];
			j--;
			goto P4;
		}
		if (q == j + 1) {
			if (j == 0)
				break;
			s++;
			d[j] = -d[j];
			j--;
			goto P4;
		}
		t = a[j - c[j] + s];
		a[j - c[j] + s] = a[j - q + s];
		a[j - q + s] = t;
		c[j] = q;
		sign = -sign;
	}
	free(a);
	stack[h] = stack[tos - 1];
	tos = h + 1;
}

//-----------------------------------------------------------------------------
//
//	Input:		Matrix on stack
//
//	Output:		Determinant on stack
//
//	Note:
//
//	Uses Gaussian elimination which is faster for numerical matrices.
//
//	Gaussian Elimination works by walking down the diagonal and clearing
//	out the columns below it.
//
//-----------------------------------------------------------------------------

void
detg(void)
{
	save();
	p1 = pop();
	if (det_check_arg() == 0) {
		push_symbol(DET);
		push(p1);
		list(2);
		restore();
		return;
	}
	yydetg();
	restore();
}

void
yydetg(void)
{
	int i, n;
	n = p1->u.tensor->dim[0];
	for (i = 0; i < n * n; i++)
		push(p1->u.tensor->elem[i]);
	lu_decomp(n);
	tos -= n * n;
	push(p1);
}

//-----------------------------------------------------------------------------
//
//	Input:		n * n matrix elements on stack
//
//	Output:		p1	determinant
//
//			p2	mangled
//
//			upper diagonal matrix on stack
//
//-----------------------------------------------------------------------------

#undef M

#define M(i, j) stack[h + n * (i) + (j)]

void
lu_decomp(int n)
{
	int d, h, i, j;
	h = tos - n * n;
	p1 = one;
	for (d = 0; d < n - 1; d++) {
		// diagonal element zero?
		if (equal(M(d, d), zero)) {
			// find a new row
			for (i = d + 1; i < n; i++)
				if (!equal(M(i, d), zero))
					break;
			if (i == n) {
				p1 = zero;
				break;
			}
			// exchange rows
			for (j = d; j < n; j++) {
				p2 = M(d, j);
				M(d, j) = M(i, j);
				M(i, j) = p2;
			}
			// negate det
			push(p1);
			negate();
			p1 = pop();
		}
		// update det
		push(p1);
		push(M(d, d));
		multiply();
		p1 = pop();
		// update lower diagonal matrix
		for (i = d + 1; i < n; i++) {
			// multiplier
			push(M(i, d));
			push(M(d, d));
			divide();
			negate();
			p2 = pop();
			// update one row
			M(i, d) = zero; // clear column below pivot d
			for (j = d + 1; j < n; j++) {
				push(M(d, j));
				push(p2);
				multiply();
				push(M(i, j));
				add();
				M(i, j) = pop();
			}
		}
	}
	// last diagonal element
	push(p1);
	push(M(n - 1, n - 1));
	multiply();
	p1 = pop();
}

//-----------------------------------------------------------------------------
//
//	Examples:
//
//	   012345678
//	-2 .........
//	-1 .........
//	 0 ..hello..	x=2, y=0, h=1, w=5
//	 1 .........
//	 2 .........
//
//	   012345678
//	-2 .........
//	-1 ..355....
//	 0 ..---....	x=2, y=-1, h=3, w=3
//	 1 ..113....
//	 2 .........
//
//-----------------------------------------------------------------------------

#undef YMAX
#define YMAX 10000

struct glyph {
	int c, x, y;
} chartab[YMAX];

int yindex, level, emit_x;
int expr_level;
int display_flag;

void
display(void)
{
	save();
	p1 = pop();
	yindex = 0;
	level = 0;
	emit_x = 0;
	emit_top_expr(p1);
	print_it();
	restore();
}

void
emit_top_expr(U *p)
{
	if (car(p) == symbol(SETQ)) {
		emit_expr(cadr(p));
		emit_str(" = ");
		emit_expr(caddr(p));
		return;
	}
	if (istensor(p))
		emit_tensor(p);
	else
		emit_expr(p);
}

int
will_be_displayed_as_fraction(U *p)
{
	if (level > 0)
		return 0;
	if (isfraction(p))
		return 1;
	if (car(p) != symbol(MULTIPLY))
		return 0;
	if (isfraction(cadr(p)))
		return 1;
	while (iscons(p)) {
		if (isdenominator(car(p)))
			return 1;
		p = cdr(p);
	}
	return 0;
}

void
emit_expr(U *p)
{
//	if (level > 0) {
//		printexpr(p);
//		return;
//	}
	expr_level++;
	if (car(p) == symbol(ADD)) {
		p = cdr(p);
		if (is_negative(car(p))) {
			emit_char('-');
			if (will_be_displayed_as_fraction(car(p)))
				emit_char(' ');
		}
		emit_term(car(p));
		p = cdr(p);
		while (iscons(p)) {
			if (is_negative(car(p))) {
				emit_char(' ');
				emit_char('-');
				emit_char(' ');
			} else {
				emit_char(' ');
				emit_char('+');
				emit_char(' ');
			}
			emit_term(car(p));
			p = cdr(p);
		}
	} else {
		if (is_negative(p)) {
			emit_char('-');
			if (will_be_displayed_as_fraction(p))
				emit_char(' ');
		}
		emit_term(p);
	}
	expr_level--;
}

void
emit_unsigned_expr(U *p)
{
	if (car(p) == symbol(ADD)) {
		p = cdr(p);
//		if (is_negative(car(p)))
//			emit_char('-');
		emit_term(car(p));
		p = cdr(p);
		while (iscons(p)) {
			if (is_negative(car(p))) {
				emit_char(' ');
				emit_char('-');
				emit_char(' ');
			} else {
				emit_char(' ');
				emit_char('+');
				emit_char(' ');
			}
			emit_term(car(p));
			p = cdr(p);
		}
	} else {
//		if (is_negative(p))
//			emit_char('-');
		emit_term(p);
	}
}

int
is_negative(U *p)
{
	if (isnegativenumber(p))
		return 1;
	if (car(p) == symbol(MULTIPLY) && isnegativenumber(cadr(p)))
		return 1;
	return 0;
}

void
emit_term(U *p)
{
	int n;
	if (car(p) == symbol(MULTIPLY)) {
		n = count_denominators(p);
		if (n && level == 0)
			emit_fraction(p, n);
		else
			emit_multiply(p, n);
	} else
		emit_factor(p);
}

int
isdenominator(U *p)
{
	if (car(p) == symbol(POWER) && cadr(p) != symbol(EXP1) && is_negative(caddr(p)))
		return 1;
	else
		return 0;
}

int
count_denominators(U *p)
{
	int count = 0;
	U *q;
	p = cdr(p);
//	if (isfraction(car(p))) {
//		count++;
//		p = cdr(p);
//	}
	while (iscons(p)) {
		q = car(p);
		if (isdenominator(q))
			count++;
		p = cdr(p);
	}
	return count;
}

// n is the number of denominators, not counting a fraction like 1/2

void
emit_multiply(U *p, int n)
{
	if (n == 0) {
		p = cdr(p);
		if (isplusone(car(p)) || isminusone(car(p)))
			p = cdr(p);
		emit_factor(car(p));
		p = cdr(p);
		while (iscons(p)) {
			emit_char(' ');
			emit_factor(car(p));
			p = cdr(p);
		}
	} else {
		emit_numerators(p);
		emit_char('/');
		// need grouping if more than one denominator
		if (n > 1 || isfraction(cadr(p))) {
			emit_char('(');
			emit_denominators(p);
			emit_char(')');
		} else
			emit_denominators(p);
	}
}

#undef A
#undef B

#define A p3
#define B p4

// sign of term has already been emitted

void
emit_fraction(U *p, int d)
{
	int count, k1, k2, n, x;
	save();
	A = one;
	B = one;
	// handle numerical coefficient
	if (isrational(cadr(p))) {
		push(cadr(p));
		mp_numerator();
		absval();
		A = pop();
		push(cadr(p));
		mp_denominator();
		B = pop();
	}
	if (isdouble(cadr(p))) {
		push(cadr(p));
		absval();
		A = pop();
	}
	// count numerators
	if (isplusone(A))
		n = 0;
	else
		n = 1;
	p1 = cdr(p);
	if (isnum(car(p1)))
		p1 = cdr(p1);
	while (iscons(p1)) {
		p2 = car(p1);
		if (isdenominator(p2))
			;
		else
			n++;
		p1 = cdr(p1);
	}
	// emit numerators
	x = emit_x;
	k1 = yindex;
	count = 0;
	// emit numerical coefficient
	if (!isplusone(A)) {
		emit_number(A, 0);
		count++;
	}
	// skip over "multiply"
	p1 = cdr(p);
	// skip over numerical coefficient, already handled
	if (isnum(car(p1)))
		p1 = cdr(p1);
	while (iscons(p1)) {
		p2 = car(p1);
		if (isdenominator(p2))
			;
		else {
			if (count > 0)
				emit_char(' ');
			if (n == 1)
				emit_expr(p2);
			else
				emit_factor(p2);
			count++;
		}
		p1 = cdr(p1);
	}
	if (count == 0)
		emit_char('1');
	// emit denominators
	k2 = yindex;
	count = 0;
	if (!isplusone(B)) {
		emit_number(B, 0);
		count++;
		d++;
	}
	p1 = cdr(p);
	if (isrational(car(p1)))
		p1 = cdr(p1);
	while (iscons(p1)) {
		p2 = car(p1);
		if (isdenominator(p2)) {
			if (count > 0)
				emit_char(' ');
			emit_denominator(p2, d);
			count++;
		}
		p1 = cdr(p1);
	}
	fixup_fraction(x, k1, k2);
	restore();
}

// p points to a multiply

void
emit_numerators(U *p)
{
	int n;
	save();
	p1 = one;
	p = cdr(p);
	if (isrational(car(p))) {
		push(car(p));
		mp_numerator();
		absval();
		p1 = pop();
		p = cdr(p);
	} else if (isdouble(car(p))) {
		push(car(p));
		absval();
		p1 = pop();
		p = cdr(p);
	}
	n = 0;
	if (!isplusone(p1)) {
		emit_number(p1, 0);
		n++;
	}
	while (iscons(p)) {
		if (isdenominator(car(p)))
			;
		else {
			if (n > 0)
				emit_char(' ');
			emit_factor(car(p));
			n++;
		}
		p = cdr(p);
	}
	if (n == 0)
		emit_char('1');
	restore();
}

// p points to a multiply

void
emit_denominators(U *p)
{
	int n;
	save();
	n = 0;
	p = cdr(p);
	if (isfraction(car(p))) {
		push(car(p));
		mp_denominator();
		p1 = pop();
		emit_number(p1, 0);
		n++;
		p = cdr(p);
	}
	while (iscons(p)) {
		if (isdenominator(car(p))) {
			if (n > 0)
				emit_char(' ');
			emit_denominator(car(p), 0);
			n++;
		}
		p = cdr(p);
	}
	restore();
}

void
emit_factor(U *p)
{
	if (istensor(p)) {
		if (level == 0)
			//emit_tensor(p);
			emit_flat_tensor(p);
		else
			emit_flat_tensor(p);
		return;
	}
	if (isdouble(p)) {
		emit_number(p, 0);
		return;
	}
	if (car(p) == symbol(ADD) || car(p) == symbol(MULTIPLY)) {
		emit_subexpr(p);
		return;
	}
	if (car(p) == symbol(POWER)) {
		emit_power(p);
		return;
	}
	if (iscons(p)) {
		//if (car(p) == symbol(FORMAL) && cadr(p)->k == SYM)
		//	emit_symbol(cadr(p));
		//else
			emit_function(p);
		return;
	}
	if (isnum(p)) {
		if (level == 0)
			emit_numerical_fraction(p);
		else
			emit_number(p, 0);
		return;
	}
	if (issymbol(p)) {
		emit_symbol(p);
		return;
	}
	if (isstr(p)) {
		emit_string(p);
		return;
	}
}

void
emit_numerical_fraction(U *p)
{
	int k1, k2, x;
	save();
	push(p);
	mp_numerator();
	absval();
	A = pop();
	push(p);
	mp_denominator();
	B = pop();
	if (isplusone(B)) {
		emit_number(A, 0);
		restore();
		return;
	}
	x = emit_x;
	k1 = yindex;
	emit_number(A, 0);
	k2 = yindex;
	emit_number(B, 0);
	fixup_fraction(x, k1, k2);
	restore();
}

// if it's a factor then it doesn't need parens around it, i.e. 1/sin(theta)^2

int
isfactor(U *p)
{
	if (iscons(p) && car(p) != symbol(ADD) && car(p) != symbol(MULTIPLY) && car(p) != symbol(POWER))
		return 1;
	if (issymbol(p))
		return 1;
	if (isfraction(p))
		return 0;
	if (isnegativenumber(p))
		return 0;
	if (isnum(p))
		return 1;
	return 0;
}

void
emit_power(U *p)
{
	int k1, k2, x;
	if (cadr(p) == symbol(EXP1)) {
		emit_str("exp(");
		emit_expr(caddr(p));
		emit_char(')');
		return;
	}
	if (level > 0) {
		if (isminusone(caddr(p))) {
			emit_char('1');
			emit_char('/');
			if (isfactor(cadr(p)))
				emit_factor(cadr(p));
			else
				emit_subexpr(cadr(p));
		} else {
			if (isfactor(cadr(p)))
				emit_factor(cadr(p));
			else
				emit_subexpr(cadr(p));
			emit_char('^');
			if (isfactor(caddr(p)))
				emit_factor(caddr(p));
			else
				emit_subexpr(caddr(p));
		}
		return;
	}
	// special case: 1 over something
	if (is_negative(caddr(p))) {
		x = emit_x;
		k1 = yindex;
		emit_char('1');
		k2 = yindex;
		//level++;
		emit_denominator(p, 1);
		//level--;
		fixup_fraction(x, k1, k2);
		return;
	}
	k1 = yindex;
	if (isfactor(cadr(p)))
		emit_factor(cadr(p));
	else
		emit_subexpr(cadr(p));
	k2 = yindex;
	level++;
	emit_expr(caddr(p));
	level--;
	fixup_power(k1, k2);
}

// if n == 1 then emit as expr (no parens)

// p is a power

void
emit_denominator(U *p, int n)
{
	int k1, k2;
	// special case: 1 over something
	if (isminusone(caddr(p))) {
		if (n == 1)
			emit_expr(cadr(p));
		else
			emit_factor(cadr(p));
		return;
	}
	k1 = yindex;
	// emit base
	if (isfactor(cadr(p)))
		emit_factor(cadr(p));
	else
		emit_subexpr(cadr(p));
	k2 = yindex;
	// emit exponent, don't emit minus sign
	level++;
	emit_unsigned_expr(caddr(p));
	level--;
	fixup_power(k1, k2);
}

void
emit_function(U *p)
{
	if (car(p) == symbol(INDEX) && issymbol(cadr(p))) {
		emit_index_function(p);
		return;
	}
	if (car(p) == symbol(FACTORIAL)) {
		emit_factorial_function(p);
		return;
	}
	if (car(p) == symbol(DERIVATIVE))
		emit_char('d');
	else
		emit_symbol(car(p));
	emit_char('(');
	p = cdr(p);
	if (iscons(p)) {
		emit_expr(car(p));
		p = cdr(p);
		while (iscons(p)) {
			emit_char(',');
			//emit_char(' ');
			emit_expr(car(p));
			p = cdr(p);
		}
	}
	emit_char(')');
}

void
emit_index_function(U *p)
{
	p = cdr(p);
	if (caar(p) == symbol(ADD) || caar(p) == symbol(MULTIPLY) || caar(p) == symbol(POWER) || caar(p) == symbol(FACTORIAL))
		emit_subexpr(car(p));
	else
		emit_expr(car(p));
	emit_char('[');
	p = cdr(p);
	if (iscons(p)) {
		emit_expr(car(p));
		p = cdr(p);
		while(iscons(p)) {
			emit_char(',');
			emit_expr(car(p));
			p = cdr(p);
		}
	}
	emit_char(']');
}

void
emit_factorial_function(U *p)
{
	p = cadr(p);
	if (car(p) == symbol(ADD) || car(p) == symbol(MULTIPLY) || car(p) == symbol(POWER) || car(p) == symbol(FACTORIAL))
		emit_subexpr(p);
	else
		emit_expr(p);
	emit_char('!');
}

void
emit_subexpr(U *p)
{
	emit_char('(');
	emit_expr(p);
	emit_char(')');
}

void
emit_symbol(U *p)
{
	char *s;
	if (p == symbol(EXP1)) {
		emit_str("exp(1)");
		return;
	}
	s = get_printname(p);
	while (*s)
		emit_char(*s++);
}

void
emit_string(U *p)
{
	char *s;
	s = p->u.str;
	while (*s)
		emit_char(*s++);
}

void
fixup_fraction(int x, int k1, int k2)
{
	int dx, dy, i, w, y;
	int h1, w1, y1;
	int h2, w2, y2;
	get_size(k1, k2, &h1, &w1, &y1);
	get_size(k2, yindex, &h2, &w2, &y2);
	if (w2 > w1)
		dx = (w2 - w1) / 2;	// shift numerator right
	else
		dx = 0;
dx++;
	// this is how much is below the baseline
	y = y1 + h1 - 1;
	dy = -y - 1;
	move(k1, k2, dx, dy);
	if (w2 > w1)
		dx = -w1;
	else
		dx = -w1 + (w1 - w2) / 2;
dx++;
	dy = -y2 + 1;
	move(k2, yindex, dx, dy);
	if (w2 > w1)
		w = w2;
	else
		w = w1;
w+=2;
	emit_x = x;
	for (i = 0; i < w; i++)
		emit_char('-');
}

void
fixup_power(int k1, int k2)
{
	int dy;
	int h1, w1, y1;
	int h2, w2, y2;
	get_size(k1, k2, &h1, &w1, &y1);
	get_size(k2, yindex, &h2, &w2, &y2);
	// move superscript to baseline
	dy = -y2 - h2 + 1;
	// now move above base
	dy += y1 - 1;
	move(k2, yindex, 0, dy);
}

void
move(int j, int k, int dx, int dy)
{
	int i;
	for (i = j; i < k; i++) {
		chartab[i].x += dx;
		chartab[i].y += dy;
	}
}

// finds the bounding rectangle and vertical position

void
get_size(int j, int k, int *h, int *w, int *y)
{
	int i;
	int min_x, max_x, min_y, max_y;
	min_x = chartab[j].x;
	max_x = chartab[j].x;
	min_y = chartab[j].y;
	max_y = chartab[j].y;
	for (i = j + 1; i < k; i++) {
		if (chartab[i].x < min_x)
			min_x = chartab[i].x;
		if (chartab[i].x > max_x)
			max_x = chartab[i].x;
		if (chartab[i].y < min_y)
			min_y = chartab[i].y;
		if (chartab[i].y > max_y)
			max_y = chartab[i].y;
	}
	*h = max_y - min_y + 1;
	*w = max_x - min_x + 1;
	*y = min_y;
}

void
displaychar(int c)
{
	emit_char(c);
}

void
emit_char(int c)
{
	if (yindex == YMAX)
		return;
	chartab[yindex].c = c;
	chartab[yindex].x = emit_x;
	chartab[yindex].y = 0;
	yindex++;
	emit_x++;
}

void
emit_str(char *s)
{
	while (*s)
		emit_char(*s++);
}

void
emit_number(U *p, int emit_sign)
{
	char *s;
	static char buf[100];
	switch (p->k) {
	case NUM:
		s = mstr(p->u.q.a);
		if (*s == '-' && emit_sign == 0)
			s++;
		while (*s)
			emit_char(*s++);
		s = mstr(p->u.q.b);
		if (strcmp(s, "1") == 0)
			break;
		emit_char('/');
		while (*s)
			emit_char(*s++);
		break;
	case DOUBLE:
		sprintf(buf, "%g", p->u.d);
		s = buf;
		if (*s == '-' && emit_sign == 0)
			s++;
		while (*s)
			emit_char(*s++);
		break;
	default:
		break;
	}
}

int
display_cmp(const void *aa, const void *bb)
{
	struct glyph *a, *b;
	a = (struct glyph *) aa;
	b = (struct glyph *) bb;
	if (a->y < b->y)
		return -1;
	if (a->y > b->y)
		return 1;
	if (a->x < b->x)
		return -1;
	if (a->x > b->x)
		return 1;
	return 0;
}

void
print_it(void)
{
	int i, x, y;
	qsort(chartab, yindex, sizeof (struct glyph), display_cmp);
	x = 0;
	y = chartab[0].y;
	for (i = 0; i < yindex; i++) {
		while (chartab[i].y > y) {
			printchar('\n');
			x = 0;
			y++;
		}
		while (chartab[i].x > x) {
			printchar_nowrap(' ');
			x++;
		}
		printchar_nowrap(chartab[i].c);
		x++;
	}
	printchar('\n');
}

char print_buffer[10000];

char *
getdisplaystr(void)
{
	yindex = 0;
	level = 0;
	emit_x = 0;
	emit_expr(pop());
	fill_buf();
	return print_buffer;
}

void
fill_buf(void)
{
	int i, k, x, y;
	qsort(chartab, yindex, sizeof (struct glyph), display_cmp);
	k = 0;
	x = 0;
	y = chartab[0].y;
	for (i = 0; i < yindex; i++) {
		while (chartab[i].y > y) {
			if (k < sizeof print_buffer - 2)
				print_buffer[k++] = '\n';
			x = 0;
			y++;
		}
		while (chartab[i].x > x) {
			if (k < sizeof print_buffer - 2)
				print_buffer[k++] = ' ';
			x++;
		}
		if (k < sizeof print_buffer - 2)
			print_buffer[k++] = chartab[i].c;
		x++;
	}
	if (k == sizeof print_buffer - 2)
		printf("warning: print buffer full\n");
	print_buffer[k++] = '\n';
	print_buffer[k++] = '\0';
}

#undef N

#define N 100

struct elem {
	int x, y, h, w, index, count;
} elem[N];

#define SPACE_BETWEEN_COLUMNS 3
#define SPACE_BETWEEN_ROWS 1

void
emit_tensor(U *p)
{
	int i, n, nrow, ncol;
	int x, y;
	int h, w;
	int dx, dy;
	int eh, ew;
	int row, col;
	if (p->u.tensor->ndim > 2) {
		emit_flat_tensor(p);
		return;
	}
	nrow = p->u.tensor->dim[0];
	if (p->u.tensor->ndim == 2)
		ncol = p->u.tensor->dim[1];
	else
		ncol = 1;
	n = nrow * ncol;
	if (n > N) {
		emit_flat_tensor(p);
		return;
	}
	// horizontal coordinate of the matrix
#if 0
	emit_x += 2; // make space for left paren
#endif
	x = emit_x;
	// emit each element
	for (i = 0; i < n; i++) {
		elem[i].index = yindex;
		elem[i].x = emit_x;
		emit_expr(p->u.tensor->elem[i]);
		elem[i].count = yindex - elem[i].index;
		get_size(elem[i].index, yindex, &elem[i].h, &elem[i].w, &elem[i].y);
	}
	// find element height and width
	eh = 0;
	ew = 0;
	for (i = 0; i < n; i++) {
		if (elem[i].h > eh)
			eh = elem[i].h;
		if (elem[i].w > ew)
			ew = elem[i].w;
	}
	// this is the overall height of the matrix
	h = nrow * eh + (nrow - 1) * SPACE_BETWEEN_ROWS;
	// this is the overall width of the matrix
	w = ncol * ew + (ncol - 1) * SPACE_BETWEEN_COLUMNS;
	// this is the vertical coordinate of the matrix
	y = -(h / 2);
	// move elements around
	for (row = 0; row < nrow; row++) {
		for (col = 0; col < ncol; col++) {
			i = row * ncol + col;
			// first move to upper left corner of matrix
			dx = x - elem[i].x;
			dy = y - elem[i].y;
			move(elem[i].index, elem[i].index + elem[i].count, dx, dy);
			// now move to official position
			dx = 0;
			if (col > 0)
				dx = col * (ew + SPACE_BETWEEN_COLUMNS);
			dy = 0;
			if (row > 0)
				dy = row * (eh + SPACE_BETWEEN_ROWS);
			// small correction for horizontal centering
			dx += (ew - elem[i].w) / 2;
			// small correction for vertical centering
			dy += (eh - elem[i].h) / 2;
			move(elem[i].index, elem[i].index + elem[i].count, dx, dy);
		}
	}
	emit_x = x + w;
#if 0
	// left brace
	for (i = 0; i < h; i++) {
		if (yindex == YMAX)
			break;
		chartab[yindex].c = '|';
		chartab[yindex].x = x - 2;
		chartab[yindex].y = y + i;
		yindex++;
	}
	// right brace
	emit_x++;
	for (i = 0; i < h; i++) {
		if (yindex == YMAX)
			break;
		chartab[yindex].c = '|';
		chartab[yindex].x = emit_x;
		chartab[yindex].y = y + i;
		yindex++;
	}
	emit_x++;
#endif
}

void
emit_flat_tensor(U *p)
{
	int k = 0;
	emit_tensor_inner(p, 0, &k);
}

void
emit_tensor_inner(U *p, int j, int *k)
{
	int i;
	emit_char('(');
	for (i = 0; i < p->u.tensor->dim[j]; i++) {
		if (j + 1 == p->u.tensor->ndim) {
			emit_expr(p->u.tensor->elem[*k]);
			*k = *k + 1;
		} else
			emit_tensor_inner(p, j + 1, k);
		if (i + 1 < p->u.tensor->dim[j])
			emit_char(',');
	}
	emit_char(')');
}

//	take expr and push all constant subexpr

//	p1	expr

//	p2	independent variable (like x)

void
distill(void)
{
	save();
	distill_nib();
	restore();
}

void
distill_nib(void)
{
	p2 = pop();
	p1 = pop();
	// is the entire expression constant?
	if (find(p1, p2) == 0) {
		push(p1);
		//push(p1);	// may need later for pushing both +a, -a
		//negate();
		return;
	}
	// sum?
	if (isadd(p1)) {
		distill_sum();
		return;
	}
	// product?
	if (car(p1) == symbol(MULTIPLY)) {
		distill_product();
		return;
	}
	// naive distill if not sum or product
	p3 = cdr(p1);
	while (iscons(p3)) {
		push(car(p3));
		push(p2);
		distill();
		p3 = cdr(p3);
	}
}

void
distill_sum(void)
{
	int h;
	// distill terms involving x
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2)) {
			push(car(p3));
			push(p2);
			distill();
		}
		p3 = cdr(p3);
	}
	// add together all constant terms
	h = tos;
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2) == 0)
			push(car(p3));
		p3 = cdr(p3);
	}
	if (tos - h) {
		add_all(tos - h);
		p3 = pop();
		push(p3);
		push(p3);
		negate();	// need both +a, -a for some integrals
	}
}

void
distill_product(void)
{
	int h;
	// distill factors involving x
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2)) {
			push(car(p3));
			push(p2);
			distill();
		}
		p3 = cdr(p3);
	}
	// multiply together all constant factors
	h = tos;
	p3 = cdr(p1);
	while (iscons(p3)) {
		if (find(car(p3), p2) == 0)
			push(car(p3));
		p3 = cdr(p3);
	}
	if (tos - h) {
		multiply_all(tos - h);
		//p3 = pop();	// may need later for pushing both +a, -a
		//push(p3);
		//push(p3);
		//negate();
	}
}

//-----------------------------------------------------------------------------
//
//	Generate all divisors of a term
//
//	Input:		Term on stack (factor * factor * ...)
//
//	Output:		Divisors on stack
//
//-----------------------------------------------------------------------------

void
divisors(void)
{
	int i, h, n;
	save();
	h = tos - 1;
	divisors_onstack();
	n = tos - h;
	qsort(stack + h, n, sizeof (U *), divisors_cmp);
	p1 = alloc_tensor(n);
	p1->u.tensor->ndim = 1;
	p1->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++)
		p1->u.tensor->elem[i] = stack[h + i];
	tos = h;
	push(p1);
	restore();
}

void
divisors_onstack(void)
{
	int h, i, k, n;
	save();
	p1 = pop();
	h = tos;
	// push all of the term's factors
	if (isnum(p1)) {
		push(p1);
		factor_small_number();
	} else if (car(p1) == symbol(ADD)) {
		push(p1);
		factor_add();
//printf(">>>\n");
//for (i = h; i < tos; i++)
//print(stdout, stack[i]);
//printf("<<<\n");
	} else if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		if (isnum(car(p1))) {
			push(car(p1));
			factor_small_number();
			p1 = cdr(p1);
		}
		while (iscons(p1)) {
			p2 = car(p1);
			if (car(p2) == symbol(POWER)) {
				push(cadr(p2));
				push(caddr(p2));
			} else {
				push(p2);
				push(one);
			}
			p1 = cdr(p1);
		}
	} else if (car(p1) == symbol(POWER)) {
		push(cadr(p1));
		push(caddr(p1));
	} else {
		push(p1);
		push(one);
	}
	k = tos;
	// contruct divisors by recursive descent
	push(one);
	gen(h, k);
	// move
	n = tos - k;
	for (i = 0; i < n; i++)
		stack[h + i] = stack[k + i];
	tos = h + n;
	restore();
}

//-----------------------------------------------------------------------------
//
//	Generate divisors
//
//	Input:		Base-exponent pairs on stack
//
//			h	first pair
//
//			k	just past last pair
//
//	Output:		Divisors on stack
//
//	For example, factor list 2 2 3 1 results in 6 divisors,
//
//		1
//		3
//		2
//		6
//		4
//		12
//
//-----------------------------------------------------------------------------

#undef ACCUM
#undef BASE
#undef EXPO

#define ACCUM p1
#define BASE p2
#define EXPO p3

void
gen(int h, int k)
{
	int expo, i;
	save();
	ACCUM = pop();
	if (h == k) {
		push(ACCUM);
		restore();
		return;
	}
	BASE = stack[h + 0];
	EXPO = stack[h + 1];
	push(EXPO);
	expo = pop_integer();
	for (i = 0; i <= abs(expo); i++) {
		push(ACCUM);
		push(BASE);
		push_integer(sign(expo) * i);
		power();
		multiply();
		gen(h + 2, k);
	}
	restore();
}

//-----------------------------------------------------------------------------
//
//	Factor ADD expression
//
//	Input:		Expression on stack
//
//	Output:		Factors on stack
//
//	Each factor consists of two expressions, the factor itself followed
//	by the exponent.
//
//-----------------------------------------------------------------------------

void
factor_add(void)
{
	save();
	p1 = pop();
	// get gcd of all terms
	p3 = cdr(p1);
	push(car(p3));
	p3 = cdr(p3);
	while (iscons(p3)) {
		push(car(p3));
		gcd();
		p3 = cdr(p3);
	}
	// check gcd
	p2 = pop();
	if (isplusone(p2)) {
		push(p1);
		push(one);
		restore();
		return;
	}
	// push factored gcd
	if (isnum(p2)) {
		push(p2);
		factor_small_number();
	} else if (car(p2) == symbol(MULTIPLY)) {
		p3 = cdr(p2);
		if (isnum(car(p3))) {
			push(car(p3));
			factor_small_number();
		} else {
			push(car(p3));
			push(one);
		}
		p3 = cdr(p3);
		while (iscons(p3)) {
			push(car(p3));
			push(one);
			p3 = cdr(p3);
		}
	} else {
		push(p2);
		push(one);
	}
	// divide each term by gcd
	push(p2);
	inverse();
	p2 = pop();
	push(zero);
	p3 = cdr(p1);
	while (iscons(p3)) {
		push(p2);
		push(car(p3));
		multiply();
		add();
		p3 = cdr(p3);
	}
	push(one);
	restore();
}

int
divisors_cmp(const void *p1, const void *p2)
{
	return cmp_expr(*((U **) p1), *((U **) p2));
}

// power function for double precision floating point

void
dpow(void)
{
	double a, b, base, expo, result, theta;
	expo = pop_double();
	base = pop_double();
	// divide by zero?
	if (base == 0.0 && expo < 0.0)
		stop("divide by zero");
	// nonnegative base or integer power?
	if (base >= 0.0 || fmod(expo, 1.0) == 0.0) {
		result = pow(base, expo);
		push_double(result);
		return;
	}
	result = pow(fabs(base), expo);
	theta = M_PI * expo;
	// this ensures the real part is 0.0 instead of a tiny fraction
	if (fmod(expo, 0.5) == 0.0) {
		a = 0.0;
		b = sin(theta);
	} else {
		a = cos(theta);
		b = sin(theta);
	}
	push_double(a * result);
	push_double(b * result);
	push(imaginaryunit);
	multiply();
	add();
}

//-----------------------------------------------------------------------------
//
//	Compute eigenvalues and eigenvectors
//
//	Input:		stack[tos - 1]		symmetric matrix
//
//	Output:		D			diagnonal matrix
//
//			Q			eigenvector matrix
//
//	D and Q have the property that
//
//		A == dot(transpose(Q),D,Q)
//
//	where A is the original matrix.
//
//	The eigenvalues are on the diagonal of D.
//
//	The eigenvectors are row vectors in Q.
//
//	The eigenvalue relation
//
//		A X = lambda X
//
//	can be checked as follows:
//
//		lambda = D[1,1]
//
//		X = Q[1]
//
//		dot(A,X) - lambda X
//
//-----------------------------------------------------------------------------

#undef D
#undef Q

#define D(i, j) yydd[eigen_n * (i) + (j)]
#define Q(i, j) yyqq[eigen_n * (i) + (j)]

int eigen_n;
double *yydd, *yyqq;

void
eval_eigen(void)
{
	if (eigen_check_arg() == 0)
		stop("eigen: argument is not a square matrix");
	eigen(EIGEN);
	p1 = usr_symbol("D");
	set_binding(p1, p2);
	p1 = usr_symbol("Q");
	set_binding(p1, p3);
	push(symbol(NIL));
}

void
eval_eigenval(void)
{
	if (eigen_check_arg() == 0) {
		push_symbol(EIGENVAL);
		push(p1);
		list(2);
		return;
	}
	eigen(EIGENVAL);
	push(p2);
}

void
eval_eigenvec(void)
{
	if (eigen_check_arg() == 0) {
		push_symbol(EIGENVEC);
		push(p1);
		list(2);
		return;
	}
	eigen(EIGENVEC);
	push(p3);
}

int
eigen_check_arg(void)
{
	int i, j;
	push(cadr(p1));
	eval();
	yyfloat();
	eval();
	p1 = pop();
	if (!istensor(p1))
		return 0;
	if (p1->u.tensor->ndim != 2 || p1->u.tensor->dim[0] != p1->u.tensor->dim[1])
		stop("eigen: argument is not a square matrix");
	eigen_n = p1->u.tensor->dim[0];
	for (i = 0; i < eigen_n; i++)
		for (j = 0; j < eigen_n; j++)
			if (!isdouble(p1->u.tensor->elem[eigen_n * i + j]))
				stop("eigen: matrix is not numerical");
	for (i = 0; i < eigen_n - 1; i++)
		for (j = i + 1; j < eigen_n; j++)
			if (fabs(p1->u.tensor->elem[eigen_n * i + j]->u.d - p1->u.tensor->elem[eigen_n * j + i]->u.d) > 1e-10)
				stop("eigen: matrix is not symmetrical");
	return 1;
}

//-----------------------------------------------------------------------------
//
//	Input:		p1		matrix
//
//	Output:		p2		eigenvalues
//
//			p3		eigenvectors
//
//-----------------------------------------------------------------------------

void
eigen(int op)
{
	int i, j;
	// malloc working vars
	yydd = (double *) malloc(eigen_n * eigen_n * sizeof (double));
	if (yydd == NULL)
		stop("malloc failure");
	yyqq = (double *) malloc(eigen_n * eigen_n * sizeof (double));
	if (yyqq == NULL)
		stop("malloc failure");
	// initialize D
	for (i = 0; i < eigen_n; i++) {
		D(i, i) = p1->u.tensor->elem[eigen_n * i + i]->u.d;
		for (j = i + 1; j < eigen_n; j++) {
			D(i, j) = p1->u.tensor->elem[eigen_n * i + j]->u.d;
			D(j, i) = p1->u.tensor->elem[eigen_n * i + j]->u.d;
		}
	}
	// initialize Q
	for (i = 0; i < eigen_n; i++) {
		Q(i, i) = 1.0;
		for (j = i + 1; j < eigen_n; j++) {
			Q(i, j) = 0.0;
			Q(j, i) = 0.0;
		}
	}
	// step up to 100 times
	for (i = 0; i < 100; i++)
		if (step() == 0)
			break;
	if (i == 100)
		printstr("\nnote: eigen did not converge\n");
	// p2 = D
	if (op == EIGEN || op == EIGENVAL) {
		push(p1);
		copy_tensor();
		p2 = pop();
		for (i = 0; i < eigen_n; i++) {
			for (j = 0; j < eigen_n; j++) {
				push_double(D(i, j));
				p2->u.tensor->elem[eigen_n * i + j] = pop();
			}
		}
	}
	// p3 = Q
	if (op == EIGEN || op == EIGENVEC) {
		push(p1);
		copy_tensor();
		p3 = pop();
		for (i = 0; i < eigen_n; i++) {
			for (j = 0; j < eigen_n; j++) {
				push_double(Q(i, j));
				p3->u.tensor->elem[eigen_n * i + j] = pop();
			}
		}
	}
	// free working vars
	free(yydd);
	free(yyqq);
}

//-----------------------------------------------------------------------------
//
//	Example: p = 1, q = 3
//
//		c	0	s	0
//
//		0	1	0	0
//	G =
//		-s	0	c	0
//
//		0	0	0	1
//
//	The effect of multiplying G times A is...
//
//	row 1 of A    = c (row 1 of A ) + s (row 3 of A )
//	          n+1                n                 n
//
//	row 3 of A    = c (row 3 of A ) - s (row 1 of A )
//	          n+1                n                 n
//
//	In terms of components the overall effect is...
//
//	row 1 = c row 1 + s row 3
//
//		A[1,1] = c A[1,1] + s A[3,1]
//
//		A[1,2] = c A[1,2] + s A[3,2]
//
//		A[1,3] = c A[1,3] + s A[3,3]
//
//		A[1,4] = c A[1,4] + s A[3,4]
//
//	row 3 = c row 3 - s row 1
//
//		A[3,1] = c A[3,1] - s A[1,1]
//
//		A[3,2] = c A[3,2] - s A[1,2]
//
//		A[3,3] = c A[3,3] - s A[1,3]
//
//		A[3,4] = c A[3,4] - s A[1,4]
//
//	                                   T
//	The effect of multiplying A times G  is...
//
//	col 1 of A    = c (col 1 of A ) + s (col 3 of A )
//	          n+1                n                 n
//
//	col 3 of A    = c (col 3 of A ) - s (col 1 of A )
//	          n+1                n                 n
//
//	In terms of components the overall effect is...
//
//	col 1 = c col 1 + s col 3
//
//		A[1,1] = c A[1,1] + s A[1,3]
//
//		A[2,1] = c A[2,1] + s A[2,3]
//
//		A[3,1] = c A[3,1] + s A[3,3]
//
//		A[4,1] = c A[4,1] + s A[4,3]
//
//	col 3 = c col 3 - s col 1
//
//		A[1,3] = c A[1,3] - s A[1,1]
//
//		A[2,3] = c A[2,3] - s A[2,1]
//
//		A[3,3] = c A[3,3] - s A[3,1]
//
//		A[4,3] = c A[4,3] - s A[4,1]
//
//	What we want to do is just compute the upper triangle of A since we
//	know the lower triangle is identical.
//
//	In other words, we just want to update components A[i,j] where i < j.
//
//-----------------------------------------------------------------------------
//
//	Example: p = 2, q = 5
//
//				p			q
//
//			j=1	j=2	j=3	j=4	j=5	j=6
//
//		i=1	.	A[1,2]	.	.	A[1,5]	.
//
//	p	i=2	A[2,1]	A[2,2]	A[2,3]	A[2,4]	A[2,5]	A[2,6]
//
//		i=3	.	A[3,2]	.	.	A[3,5]	.
//
//		i=4	.	A[4,2]	.	.	A[4,5]	.
//
//	q	i=5	A[5,1]	A[5,2]	A[5,3]	A[5,4]	A[5,5]	A[5,6]
//
//		i=6	.	A[6,2]	.	.	A[6,5]	.
//
//-----------------------------------------------------------------------------
//
//	This is what B = GA does:
//
//	row 2 = c row 2 + s row 5
//
//		B[2,1] = c * A[2,1] + s * A[5,1]
//		B[2,2] = c * A[2,2] + s * A[5,2]
//		B[2,3] = c * A[2,3] + s * A[5,3]
//		B[2,4] = c * A[2,4] + s * A[5,4]
//		B[2,5] = c * A[2,5] + s * A[5,5]
//		B[2,6] = c * A[2,6] + s * A[5,6]
//
//	row 5 = c row 5 - s row 2
//
//		B[5,1] = c * A[5,1] + s * A[2,1]
//		B[5,2] = c * A[5,2] + s * A[2,2]
//		B[5,3] = c * A[5,3] + s * A[2,3]
//		B[5,4] = c * A[5,4] + s * A[2,4]
//		B[5,5] = c * A[5,5] + s * A[2,5]
//		B[5,6] = c * A[5,6] + s * A[2,6]
//
//	               T
//	This is what BG  does:
//
//	col 2 = c col 2 + s col 5
//
//		B[1,2] = c * A[1,2] + s * A[1,5]
//		B[2,2] = c * A[2,2] + s * A[2,5]
//		B[3,2] = c * A[3,2] + s * A[3,5]
//		B[4,2] = c * A[4,2] + s * A[4,5]
//		B[5,2] = c * A[5,2] + s * A[5,5]
//		B[6,2] = c * A[6,2] + s * A[6,5]
//
//	col 5 = c col 5 - s col 2
//
//		B[1,5] = c * A[1,5] - s * A[1,2]
//		B[2,5] = c * A[2,5] - s * A[2,2]
//		B[3,5] = c * A[3,5] - s * A[3,2]
//		B[4,5] = c * A[4,5] - s * A[4,2]
//		B[5,5] = c * A[5,5] - s * A[5,2]
//		B[6,5] = c * A[6,5] - s * A[6,2]
//
//-----------------------------------------------------------------------------
//
//	Step 1: Just do upper triangle (i < j), B[2,5] = 0
//
//		B[1,2] = c * A[1,2] + s * A[1,5]
//
//		B[2,3] = c * A[2,3] + s * A[5,3]
//		B[2,4] = c * A[2,4] + s * A[5,4]
//		B[2,6] = c * A[2,6] + s * A[5,6]
//
//		B[1,5] = c * A[1,5] - s * A[1,2]
//		B[3,5] = c * A[3,5] - s * A[3,2]
//		B[4,5] = c * A[4,5] - s * A[4,2]
//
//		B[5,6] = c * A[5,6] + s * A[2,6]
//
//-----------------------------------------------------------------------------
//
//	Step 2: Transpose where i > j since A[i,j] == A[j,i]
//
//		B[1,2] = c * A[1,2] + s * A[1,5]
//
//		B[2,3] = c * A[2,3] + s * A[3,5]
//		B[2,4] = c * A[2,4] + s * A[4,5]
//		B[2,6] = c * A[2,6] + s * A[5,6]
//
//		B[1,5] = c * A[1,5] - s * A[1,2]
//		B[3,5] = c * A[3,5] - s * A[2,3]
//		B[4,5] = c * A[4,5] - s * A[2,4]
//
//		B[5,6] = c * A[5,6] + s * A[2,6]
//
//-----------------------------------------------------------------------------
//
//	Step 3: Same as above except reorder
//
//	k < p		(k = 1)
//
//		A[1,2] = c * A[1,2] + s * A[1,5]
//		A[1,5] = c * A[1,5] - s * A[1,2]
//
//	p < k < q	(k = 3..4)
//
//		A[2,3] = c * A[2,3] + s * A[3,5]
//		A[3,5] = c * A[3,5] - s * A[2,3]
//
//		A[2,4] = c * A[2,4] + s * A[4,5]
//		A[4,5] = c * A[4,5] - s * A[2,4]
//
//	q < k		(k = 6)
//
//		A[2,6] = c * A[2,6] + s * A[5,6]
//		A[5,6] = c * A[5,6] - s * A[2,6]
//
//-----------------------------------------------------------------------------

int
step(void)
{
	int count, i, j;
	count = 0;
	// for each upper triangle "off-diagonal" component do step2
	for (i = 0; i < eigen_n - 1; i++) {
		for (j = i + 1; j < eigen_n; j++) {
			if (D(i, j) != 0.0) {
				step2(i, j);
				count++;
			}
		}
	}
	return count;
}

void
step2(int p, int q)
{
	int k;
	double t, theta;
	double c, cc, s, ss;
	// compute c and s
	// from Numerical Recipes (except they have a_qq - a_pp)
	theta = 0.5 * (D(p, p) - D(q, q)) / D(p, q);
	t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
	if (theta < 0.0)
		t = -t;
	c = 1.0 / sqrt(t * t + 1.0);
	s = t * c;
	// D = GD
	// which means "add rows"
	for (k = 0; k < eigen_n; k++) {
		cc = D(p, k);
		ss = D(q, k);
		D(p, k) = c * cc + s * ss;
		D(q, k) = c * ss - s * cc;
	}
	// D = D transpose(G)
	// which means "add columns"
	for (k = 0; k < eigen_n; k++) {
		cc = D(k, p);
		ss = D(k, q);
		D(k, p) = c * cc + s * ss;
		D(k, q) = c * ss - s * cc;
	}
	// Q = GQ
	// which means "add rows"
	for (k = 0; k < eigen_n; k++) {
		cc = Q(p, k);
		ss = Q(q, k);
		Q(p, k) = c * cc + s * ss;
		Q(q, k) = c * ss - s * cc;
	}
	D(p, q) = 0.0;
	D(q, p) = 0.0;
}

void
eval_erf(void)
{
	push(cadr(p1));
	eval();
	yerf();
}

void
yerf(void)
{
	save();
	yyerf();
	restore();
}

void
yyerf(void)
{
	double d;
	p1 = pop();
	if (isdouble(p1)) {
		d = 1.0 - erfc(p1->u.d);
		push_double(d);
		return;
	}
	if (isnegativeterm(p1)) {
		push_symbol(ERF);
		push(p1);
		negate();
		list(2);
		negate();
		return;
	}
	push_symbol(ERF);
	push(p1);
	list(2);
	return;
}

void
eval_erfc(void)
{
	push(cadr(p1));
	eval();
	yerfc();
}

void
yerfc(void)
{
	save();
	yyerfc();
	restore();
}

void
yyerfc(void)
{
	double d;
	p1 = pop();
	if (isdouble(p1)) {
		d = erfc(p1->u.d);
		push_double(d);
		return;
	}
	push_symbol(ERFC);
	push(p1);
	list(2);
}

// Evaluate an expression, for example...
//
//	push(p1)
//	eval()
//	p2 = pop()

void
eval(void)
{
	check_esc_flag();
	save();
	p1 = pop();
	switch (p1->k) {
	case CONS:
		eval_cons();
		break;
	case NUM:
		push(p1);
		break;
	case DOUBLE:
		push(p1);
		break;
	case STR:
		push(p1);
		break;
	case TENSOR:
		eval_tensor();
		break;
	case SYM:
		eval_sym();
		break;
	default:
		stop("atom?");
		break;
	}
	restore();
}

void
eval_sym(void)
{
	// bare keyword?
	if (iskeyword(p1)) {
		push(p1);
		push(symbol(LAST));
		list(2);
		eval();
		return;
	}
	// evaluate symbol's binding
	p2 = get_binding(p1);
	push(p2);
	if (p1 != p2)
		eval();
}

void
eval_cons(void)
{
	if (!issymbol(car(p1)))
		stop("cons?");
	switch (symnum(car(p1))) {
	case ABS:		eval_abs();		break;
	case ADD:		eval_add();		break;
	case ADJ:		eval_adj();		break;
	case AND:		eval_and();		break;
	case ARCCOS:		eval_arccos();		break;
	case ARCCOSH:		eval_arccosh();		break;
	case ARCSIN:		eval_arcsin();		break;
	case ARCSINH:		eval_arcsinh();		break;
	case ARCTAN:		eval_arctan();		break;
	case ARCTANH:		eval_arctanh();		break;
	case ARG:		eval_arg();		break;
	case ATOMIZE:		eval_atomize();		break;
	case BESSELJ:		eval_besselj();		break;
	case BESSELY:		eval_bessely();		break;
	case BINDING:		eval_binding();		break;
	case BINOMIAL:		eval_binomial();	break;
	case CEILING:		eval_ceiling();		break;
	case CHECK:		eval_check();		break;
	case CHOOSE:		eval_choose();		break;
	case CIRCEXP:		eval_circexp();		break;
	case CLOCK:		eval_clock();		break;
	case COEFF:		eval_coeff();		break;
	case COFACTOR:		eval_cofactor();	break;
	case CONDENSE:		eval_condense();	break;
	case CONJ:		eval_conj();		break;
	case CONTRACT:		eval_contract();	break;
	case COS:		eval_cos();		break;
	case COSH:		eval_cosh();		break;
	case DECOMP:		eval_decomp();		break;
	case DEGREE:		eval_degree();		break;
	case DEFINT:		eval_defint();		break;
	case DENOMINATOR:	eval_denominator();	break;
	case DERIVATIVE:	eval_derivative();	break;
	case DET:		eval_det();		break;
	case DIM:		eval_dim();		break;
	case DISPLAY:		eval_print();		break;
	case DIVISORS:		eval_divisors();	break;
	case DO:		eval_do();		break;
	case DOT:		eval_inner();		break;
	case DRAW:		eval_draw();		break;
	case EIGEN:		eval_eigen();		break;
	case EIGENVAL:		eval_eigenval();	break;
	case EIGENVEC:		eval_eigenvec();	break;
	case ERF:		eval_erf();		break;
	case ERFC:		eval_erfc();		break;
	case EVAL:		eval_eval();		break;
	case EXP:		eval_exp();		break;
	case EXPAND:		eval_expand();		break;
	case EXPCOS:		eval_expcos();		break;
	case EXPSIN:		eval_expsin();		break;
	case FACTOR:		eval_factor();		break;
	case FACTORIAL:		eval_factorial();	break;
	case FACTORPOLY:	eval_factorpoly();	break;
	case FILTER:		eval_filter();		break;
	case FLOATF:		eval_float();		break;
	case FLOOR:		eval_floor();		break;
	case FOR:		eval_for();		break;
	case GCD:		eval_gcd();		break;
	case HERMITE:		eval_hermite();		break;
	case HILBERT:		eval_hilbert();		break;
	case IMAG:		eval_imag();		break;
	case INDEX:		eval_index();		break;
	case INNER:		eval_inner();		break;
	case INTEGRAL:		eval_integral();	break;
	case INV:		eval_inv();		break;
	case INVG:		eval_invg();		break;
	case ISINTEGER:		eval_isinteger();	break;
	case ISPRIME:		eval_isprime();		break;
	case LAGUERRE:		eval_laguerre();	break;
	case LCM:		eval_lcm();		break;
	case LEADING:		eval_leading();		break;
	case LEGENDRE:		eval_legendre();	break;
	case LOG:		eval_log();		break;
	case MAG:		eval_mag();		break;
	case MOD:		eval_mod();		break;
	case MULTIPLY:		eval_multiply();	break;
	case NOT:		eval_not();		break;
	case NROOTS:		eval_nroots();		break;
	case NUMBER:		eval_number();		break;
	case NUMERATOR:		eval_numerator();	break;
	case OPERATOR:		eval_operator();	break;
	case OR:		eval_or();		break;
	case OUTER:		eval_outer();		break;
	case POLAR:		eval_polar();		break;
	case POWER:		eval_power();		break;
	case PRIME:		eval_prime();		break;
	case PRINT:		eval_print();		break;
	case PRODUCT:		eval_product();		break;
	case QUOTE:		eval_quote();		break;
	case QUOTIENT:		eval_quotient();	break;
	case RANK:		eval_rank();		break;
	case RATIONALIZE:	eval_rationalize();	break;
	case REAL:		eval_real();		break;
	case YYRECT:		eval_rect();		break;
	case ROOTS:		eval_roots();		break;
	case SETQ:		eval_setq();		break;
	case SGN:		eval_sgn();		break;
	case SIMPLIFY:		eval_simplify();	break;
	case SIN:		eval_sin();		break;
	case SINH:		eval_sinh();		break;
	case SQRT:		eval_sqrt();		break;
	case STOP:		eval_stop();		break;
	case SUBST:		eval_subst();		break;
	case SUM:		eval_sum();		break;
	case TAN:		eval_tan();		break;
	case TANH:		eval_tanh();		break;
	case TAYLOR:		eval_taylor();		break;
	case TEST:		eval_test();		break;
	case TESTEQ:		eval_testeq();		break;
	case TESTGE:		eval_testge();		break;
	case TESTGT:		eval_testgt();		break;
	case TESTLE:		eval_testle();		break;
	case TESTLT:		eval_testlt();		break;
	case TRANSPOSE:		eval_transpose();	break;
	case UNIT:		eval_unit();		break;
	case ZERO:		eval_zero();		break;
	default:		eval_user_function();	break;
	}
}

void
eval_binding(void)
{
	push(get_binding(cadr(p1)));
}

// checks a predicate, i.e. check(A = B)

void
eval_check(void)
{
	push(cadr(p1));
	eval_predicate();
	p1 = pop();
	if (iszero(p1))
		stop("check(arg): arg is zero");
	push(symbol(NIL)); // no result is printed
}

void
eval_det(void)
{
	push(cadr(p1));
	eval();
	det();
}

void
eval_dim(void)
{
	int n;
	push(cadr(p1));
	eval();
	p2 = pop();
	if (iscons(cddr(p1))) {
		push(caddr(p1));
		eval();
		n = pop_integer();
	} else
		n = 1;
	if (!istensor(p2))
		push_integer(1); // dim of scalar is 1
	else if (n < 1 || n > p2->u.tensor->ndim)
		push(p1);
	else
		push_integer(p2->u.tensor->dim[n - 1]);
}

void
eval_divisors(void)
{
	push(cadr(p1));
	eval();
	divisors();
}

void
eval_do(void)
{
	push(car(p1));
	p1 = cdr(p1);
	while (iscons(p1)) {
		pop();
		push(car(p1));
		eval();
		p1 = cdr(p1);
	}
}

// for example, eval(f,x,2)

void
eval_eval(void)
{
	push(cadr(p1));
	eval();
	p1 = cddr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		push(cadr(p1));
		eval();
		subst();
		p1 = cddr(p1);
	}
	eval();
}

void
eval_exp(void)
{
	push(cadr(p1));
	eval();
	exponential();
}

void
eval_factorial(void)
{
	push(cadr(p1));
	eval();
	factorial();
}

void
eval_factorpoly(void)
{
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p1 = cdr(p1);
	push(car(p1));
	eval();
	factorpoly();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		factorpoly();
		p1 = cdr(p1);
	}
}

void
eval_hermite(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	hermite();
}

void
eval_hilbert(void)
{
	push(cadr(p1));
	eval();
	hilbert();
}

void
eval_index(void)
{
	int h;
	h = tos;
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		p1 = cdr(p1);
	}
	index_function(tos - h);
}

void
eval_inv(void)
{
	push(cadr(p1));
	eval();
	inv();
}

void
eval_invg(void)
{
	push(cadr(p1));
	eval();
	invg();
}

void
eval_isinteger(void)
{
	int n;
	push(cadr(p1));
	eval();
	p1 = pop();
	if (isrational(p1)) {
		if (isinteger(p1))
			push(one);
		else
			push(zero);
		return;
	}
	if (isdouble(p1)) {
		n = (int) p1->u.d;
		if (n == p1->u.d)
			push(one);
		else
			push(zero);
		return;
	}
	push_symbol(ISINTEGER);
	push(p1);
	list(2);
}

void
eval_multiply(void)
{
	push(cadr(p1));
	eval();
	p1 = cddr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		multiply();
		p1 = cdr(p1);
	}
}

void
eval_number(void)
{
	push(cadr(p1));
	eval();
	p1 = pop();
	if (p1->k == NUM || p1->k == DOUBLE)
		push_integer(1);
	else
		push_integer(0);
}

void
eval_operator(void)
{
	int h = tos;
	push_symbol(OPERATOR);
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		p1 = cdr(p1);
	}
	list(tos - h);
}

void
eval_print(void)
{
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval_and_print_result(0);
		p1 = cdr(p1);
	}
	push(symbol(NIL));
}

void
eval_quote(void)
{
	push(cadr(p1));
}

void
eval_rank(void)
{
	push(cadr(p1));
	eval();
	p1 = pop();
	if (istensor(p1))
		push_integer(p1->u.tensor->ndim);
	else
		push(zero);
}

//-----------------------------------------------------------------------------
//
//	Example: a[1] = b
//
//	p1	*-------*-----------------------*
//		|	|			|
//		setq	*-------*-------*	b
//			|	|	|
//			index	a	1
//
//	cadadr(p1) -> a
//
//-----------------------------------------------------------------------------

void
setq_indexed(void)
{
	int h;
	p4 = cadadr(p1);
	if (!issymbol(p4))
		stop("indexed assignment: error in symbol");
	h = tos;
	push(caddr(p1));
	eval();
	p2 = cdadr(p1);
	while (iscons(p2)) {
		push(car(p2));
		eval();
		p2 = cdr(p2);
	}
	set_component(tos - h);
	p3 = pop();
	set_binding(p4, p3);
	push(symbol(NIL));
}

void
eval_setq(void)
{
	if (caadr(p1) == symbol(INDEX)) {
		setq_indexed();
		return;
	}
	if (iscons(cadr(p1))) {
		define_user_function();
		return;
	}
	if (!issymbol(cadr(p1)))
		stop("symbol assignment: error in symbol");
	push(caddr(p1));
	eval();
	p2 = pop();
	set_binding(cadr(p1), p2);
	push(symbol(NIL));
}

void
eval_sqrt(void)
{
	push(cadr(p1));
	eval();
	push_rational(1, 2);
	power();
}

void
eval_stop(void)
{
	stop("user stop");
}

void
eval_subst(void)
{
	push(cadddr(p1));
	eval();
	push(caddr(p1));
	eval();
	push(cadr(p1));
	eval();
	subst();
	eval(); // normalize
}

void
eval_unit(void)
{
	int i, n;
	push(cadr(p1));
	eval();
	n = pop_integer();
	if (n < 2) {
		push(p1);
		return;
	}
	p1 = alloc_tensor(n * n);
	p1->u.tensor->ndim = 2;
	p1->u.tensor->dim[0] = n;
	p1->u.tensor->dim[1] = n;
	for (i = 0; i < n; i++)
		p1->u.tensor->elem[n * i + i] = one;
	push(p1);
}

void
eval_noexpand(void)
{
	int x = expanding;
	expanding = 0;
	eval();
	expanding = x;
}

// like eval() except "=" is evaluated as "=="

void
eval_predicate(void)
{
	save();
	p1 = pop();
	if (car(p1) == symbol(SETQ))
		eval_testeq();
	else {
		push(p1);
		eval();
	}
	restore();
}

void
eval_and_print_result(int update)
{
	save();
	p1 = pop();
	push(p1);
	eval();
	p2 = pop();
	// "draw", "for" and "setq" return "nil", there is no result to print
	if (p2 == symbol(NIL)) {
		restore();
		return;
	}
	if (!iszero(get_binding(symbol(BAKE)))) {
		push(p2);
		bake();
		p2 = pop();
	}
	if (update)
		set_binding(symbol(LAST), p2);
	// print string result in small font
	if (isstr(p2)) {
		printstr(p2->u.str);
		printstr("\n");
		restore();
		return;
	}
	// If we evaluated the symbol "i" or "j" and the result was sqrt(-1)
	// then don't do anything.
	// Otherwise if "j" is an imaginary unit then subst.
	// Otherwise if "i" is an imaginary unit then subst.
	if ((p1 == symbol(SYMBOL_I) || p1 == symbol(SYMBOL_J)) && isimaginaryunit(p2))
		;
	else if (isimaginaryunit(get_binding(symbol(SYMBOL_J)))) {
		push(p2);
		push(imaginaryunit);
		push_symbol(SYMBOL_J);
		subst();
		p2 = pop();
	} else if (isimaginaryunit(get_binding(symbol(SYMBOL_I)))) {
		push(p2);
		push(imaginaryunit);
		push_symbol(SYMBOL_I);
		subst();
		p2 = pop();
	}
	// if we evaluated the symbol "a" and got "b" then print "a=b"
	// do not print "a=a"
	if (issymbol(p1) && !iskeyword(p1) && p1 != p2) {
		push(symbol(SETQ));
		push(p1);
		push(p2);
		list(3);
		p2 = pop();
	}
	if (equaln(get_binding(symbol(TTY)), 1))
		print(p2);
	else {
		push(p2);
		cmdisplay();
	}
	restore();
}

// Partial fraction expansion
//
// Example
//
//      expand(1/(x^3+x^2),x)
//
//        1      1       1
//      ---- - --- + -------
//        2     x     x + 1
//       x

void
eval_expand(void)
{
	// 1st arg
	push(cadr(p1));
	eval();
	// 2nd arg
	push(caddr(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		guess();
	else
		push(p2);
	expand();
}

#undef A
#undef B
#undef C
#undef F
#undef P
#undef Q
#undef T
#undef X

#define A p2
#define B p3
#define C p4
#define F p5
#define P p6
#define Q p7
#define T p8
#define X p9

void
expand(void)
{
	save();
	X = pop();
	F = pop();
	if (istensor(F)) {
		expand_tensor();
		restore();
		return;
	}
	// if sum of terms then sum over the expansion of each term
	if (car(F) == symbol(ADD)) {
		push_integer(0);
		p1 = cdr(F);
		while (iscons(p1)) {
			push(car(p1));
			push(X);
			expand();
			add();
			p1 = cdr(p1);
		}
		restore();
		return;
	}
	// B = numerator
	push(F);
	numerator();
	B = pop();
	// A = denominator
	push(F);
	denominator();
	A = pop();
	remove_negative_exponents();
	// Q = quotient
	push(B);
	push(A);
	push(X);
	divpoly();
	Q = pop();
	// remainder B = B - A * Q
	push(B);
	push(A);
	push(Q);
	multiply();
	subtract();
	B = pop();
	// if the remainder is zero then we're done
	if (iszero(B)) {
		push(Q);
		restore();
		return;
	}
	// A = factor(A)
	push(A);
	push(X);
	factorpoly();
	A = pop();
	expand_get_C();
	expand_get_B();
	expand_get_A();
	if (istensor(C)) {
		push(C);
		inv();
		push(B);
		inner();
		push(A);
		inner();
	} else {
		push(B);
		push(C);
		divide();
		push(A);
		multiply();
	}
	push(Q);
	add();
	restore();
}

void
expand_tensor(void)
{
	int i;
	push(F);
	copy_tensor();
	F = pop();
	for (i = 0; i < F->u.tensor->nelem; i++) {
		push(F->u.tensor->elem[i]);
		push(X);
		expand();
		F->u.tensor->elem[i] = pop();
	}
	push(F);
}

void
remove_negative_exponents(void)
{
	int h, i, j, k, n;
	h = tos;
	factors(A);
	factors(B);
	n = tos - h;
	// find the smallest exponent
	j = 0;
	for (i = 0; i < n; i++) {
		p1 = stack[h + i];
		if (car(p1) != symbol(POWER))
			continue;
		if (cadr(p1) != X)
			continue;
		push(caddr(p1));
		k = pop_integer();
		if (k == (int) 0x80000000)
			continue;
		if (k < j)
			j = k;
	}
	tos = h;
	if (j == 0)
		return;
	// A = A / X^j
	push(A);
	push(X);
	push_integer(-j);
	power();
	multiply();
	A = pop();
	// B = B / X^j
	push(B);
	push(X);
	push_integer(-j);
	power();
	multiply();
	B = pop();
}

// Returns the expansion coefficient matrix C.
//
// Example:
//
//       B         1
//      --- = -----------
//       A      2
//             x (x + 1)
//
// We have
//
//       B     Y1     Y2      Y3
//      --- = ---- + ---- + -------
//       A      2     x      x + 1
//             x
//
// Our task is to solve for the unknowns Y1, Y2, and Y3.
//
// Multiplying both sides by A yields
//
//           AY1     AY2      AY3
//      B = ----- + ----- + -------
//            2      x       x + 1
//           x
//
// Let
//
//            A               A                 A
//      W1 = ----       W2 = ---        W3 = -------
//             2              x               x + 1
//            x
//
// Then the coefficient matrix C is
//
//              coeff(W1,x,0)   coeff(W2,x,0)   coeff(W3,x,0)
//
//       C =    coeff(W1,x,1)   coeff(W2,x,1)   coeff(W3,x,1)
//
//              coeff(W1,x,2)   coeff(W2,x,2)   coeff(W3,x,2)
//
// It follows that
//
//       coeff(B,x,0)     Y1
//
//       coeff(B,x,1) = C Y2
//
//       coeff(B,x,2) =   Y3
//
// Hence
//
//       Y1       coeff(B,x,0)
//             -1
//       Y2 = C   coeff(B,x,1)
//
//       Y3       coeff(B,x,2)

void
expand_get_C(void)
{
	int h, i, j, n;
	U **a;
	h = tos;
	if (car(A) == symbol(MULTIPLY)) {
		p1 = cdr(A);
		while (iscons(p1)) {
			F = car(p1);
			expand_get_CF();
			p1 = cdr(p1);
		}
	} else {
		F = A;
		expand_get_CF();
	}
	n = tos - h;
	if (n == 1) {
		C = pop();
		return;
	}
	C = alloc_tensor(n * n);
	C->u.tensor->ndim = 2;
	C->u.tensor->dim[0] = n;
	C->u.tensor->dim[1] = n;
	a = stack + h;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			push(a[j]);
			push(X);
			push_integer(i);
			power();
			divide();
			push(X);
			filter();
			C->u.tensor->elem[n * i + j] = pop();
		}
	}
	tos -= n;
}

// The following table shows the push order for simple roots, repeated roots,
// and inrreducible factors.
//
//  Factor F        Push 1st        Push 2nd         Push 3rd      Push 4th
//
//
//                   A
//  x               ---
//                   x
//
//
//   2               A               A
//  x               ----            ---
//                    2              x
//                   x
//
//
//                     A
//  x + 1           -------
//                   x + 1
//
//
//         2            A              A
//  (x + 1)         ----------      -------
//                          2        x + 1
//                   (x + 1)
//
//
//   2                   A               Ax
//  x  + x + 1      ------------    ------------
//                    2               2
//                   x  + x + 1      x  + x + 1
//
//
//    2         2          A              Ax              A             Ax
//  (x  + x + 1)    --------------- ---------------  ------------  ------------
//                     2         2     2         2     2             2
//                   (x  + x + 1)    (x  + x + 1)     x  + x + 1    x  + x + 1
//
//
// For T = A/F and F = P^N we have
//
//
//      Factor F          Push 1st    Push 2nd    Push 3rd    Push 4th
//
//      x                 T
//
//       2
//      x                 T           TP
//
//
//      x + 1             T
//
//             2
//      (x + 1)           T           TP
//
//       2
//      x  + x + 1        T           TX
//
//        2         2
//      (x  + x + 1)      T           TX          TP          TPX
//
//
// Hence we want to push in the order
//
//      T * (P ^ i) * (X ^ j)
//
// for all i, j such that
//
//      i = 0, 1, ..., N - 1
//
//      j = 0, 1, ..., deg(P) - 1
//
// where index j runs first.

void
expand_get_CF(void)
{	int d, i, j, n;
	if (!find(F, X))
		return;
	trivial_divide();
	if (car(F) == symbol(POWER)) {
		push(caddr(F));
		n = pop_integer();
		P = cadr(F);
	} else {
		n = 1;
		P = F;
	}
	push(P);
	push(X);
	degree();
	d = pop_integer();
	for (i = 0; i < n; i++) {
		for (j = 0; j < d; j++) {
			push(T);
			push(P);
			push_integer(i);
			power();
			multiply();
			push(X);
			push_integer(j);
			power();
			multiply();
		}
	}
}

// Returns T = A/F where F is a factor of A.

void
trivial_divide(void)
{
	int h;
	if (car(A) == symbol(MULTIPLY)) {
		h = tos;
		p0 = cdr(A);
		while (iscons(p0)) {
			if (!equal(car(p0), F)) {
				push(car(p0));
				eval(); // force expansion of (x+1)^2, f.e.
			}
			p0 = cdr(p0);
		}
		multiply_all(tos - h);
	} else
		push_integer(1);
	T = pop();
}

// Returns the expansion coefficient vector B.

void
expand_get_B(void)
{
	int i, n;
	if (!istensor(C))
		return;
	n = C->u.tensor->dim[0];
	T = alloc_tensor(n);
	T->u.tensor->ndim = 1;
	T->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++) {
		push(B);
		push(X);
		push_integer(i);
		power();
		divide();
		push(X);
		filter();
		T->u.tensor->elem[i] = pop();
	}
	B = T;
}

// Returns the expansion fractions in A.

void
expand_get_A(void)
{
	int h, i, n;
	if (!istensor(C)) {
		push(A);
		reciprocate();
		A = pop();
		return;
	}
	h = tos;
	if (car(A) == symbol(MULTIPLY)) {
		T = cdr(A);
		while (iscons(T)) {
			F = car(T);
			expand_get_AF();
			T = cdr(T);
		}
	} else {
		F = A;
		expand_get_AF();
	}
	n = tos - h;
	T = alloc_tensor(n);
	T->u.tensor->ndim = 1;
	T->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++)
		T->u.tensor->elem[i] = stack[h + i];
	tos = h;
	A = T;
}

void
expand_get_AF(void)
{	int d, i, j, n = 1;
	if (!find(F, X))
		return;
	if (car(F) == symbol(POWER)) {
		push(caddr(F));
		n = pop_integer();
		F = cadr(F);
	}
	push(F);
	push(X);
	degree();
	d = pop_integer();
	for (i = n; i > 0; i--) {
		for (j = 0; j < d; j++) {
			push(F);
			push_integer(i);
			power();
			reciprocate();
			push(X);
			push_integer(j);
			power();
			multiply();
		}
	}
}

// exponential cosine function

void
eval_expcos(void)
{
	push(cadr(p1));
	eval();
	expcos();
}

void
expcos(void)
{
	save();
	p1 = pop();
	push(imaginaryunit);
	push(p1);
	multiply();
	exponential();
	push_rational(1, 2);
	multiply();
	push(imaginaryunit);
	negate();
	push(p1);
	multiply();
	exponential();
	push_rational(1, 2);
	multiply();
	add();
	restore();
}

// exponential sine function

void
eval_expsin(void)
{
	push(cadr(p1));
	eval();
	expsin();
}

void
expsin(void)
{
	save();
	p1 = pop();
	push(imaginaryunit);
	push(p1);
	multiply();
	exponential();
	push(imaginaryunit);
	divide();
	push_rational(1, 2);
	multiply();
	push(imaginaryunit);
	negate();
	push(p1);
	multiply();
	exponential();
	push(imaginaryunit);
	divide();
	push_rational(1, 2);
	multiply();
	subtract();
	restore();
}

// factor a polynomial or integer

void
eval_factor(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		guess();
	else
		push(p2);
	factor();
	// more factoring?
	p1 = cdddr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		factor_again();
		p1 = cdr(p1);
	}
}

void
factor_again(void)
{
	int h, n;
	save();
	p2 = pop();
	p1 = pop();
	h = tos;
	if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			push(p2);
			factor_term();
			p1 = cdr(p1);
		}
	} else {
		push(p1);
		push(p2);
		factor_term();
	}
	n = tos - h;
	if (n > 1)
		multiply_all_noexpand(n);
	restore();
}

void
factor_term(void)
{
	save();
	factorpoly();
	p1 = pop();
	if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			p1 = cdr(p1);
		}
	} else
		push(p1);
	restore();
}

void
factor(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (isinteger(p1)) {
		push(p1);
		factor_number(); // see pollard.cpp
	} else {
		push(p1);
		push(p2);
		factorpoly();
	}
	restore();
}

// for factoring small integers (2^32 or less)

void
factor_small_number(void)
{
	int d, expo, i, n;
	save();
	n = pop_integer();
	if (n == (int) 0x80000000)
		stop("number too big to factor");
	if (n < 0)
		n = -n;
	for (i = 0; i < MAXPRIMETAB; i++) {
		d = primetab[i];
		if (d > n / d)
			break;
		expo = 0;
		while (n % d == 0) {
			n /= d;
			expo++;
		}
		if (expo) {
			push_integer(d);
			push_integer(expo);
		}
	}
	if (n > 1) {
		push_integer(n);
		push_integer(1);
	}
	restore();
}

extern void bignum_factorial(int);

void
factorial(void)
{
	int n;
	save();
	p1 = pop();
	push(p1);
	n = pop_integer();
	if (n < 0 || n == (int) 0x80000000) {
		push_symbol(FACTORIAL);
		push(p1);
		list(2);
		restore();
		return;
	}
	bignum_factorial(n);
	restore();
}

void sfac_product(void);
void sfac_product_f(U **, int, int);

// simplification rules for factorials (m < n)
//
//	(e + 1) * factorial(e)	->	factorial(e + 1)
//
//	factorial(e) / e	->	factorial(e - 1)
//
//	e / factorial(e)	->	1 / factorial(e - 1)
//
//	factorial(e + n)
//	----------------	->	(e + m + 1)(e + m + 2)...(e + n)
//	factorial(e + m)
//
//	factorial(e + m)                               1
//	----------------	->	--------------------------------
//	factorial(e + n)		(e + m + 1)(e + m + 2)...(e + n)

void
simplifyfactorials(void)
{
	int x;
	save();
	x = expanding;
	expanding = 0;
	p1 = pop();
	if (car(p1) == symbol(ADD)) {
		push(zero);
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			simplifyfactorials();
			add();
			p1 = cdr(p1);
		}
		expanding = x;
		restore();
		return;
	}
	if (car(p1) == symbol(MULTIPLY)) {
		sfac_product();
		expanding = x;
		restore();
		return;
	}
	push(p1);
	expanding = x;
	restore();
}

void
sfac_product(void)
{
	int i, j, n;
	U **s;
	s = stack + tos;
	p1 = cdr(p1);
	n = 0;
	while (iscons(p1)) {
		push(car(p1));
		p1 = cdr(p1);
		n++;
	}
	for (i = 0; i < n - 1; i++) {
		if (s[i] == symbol(NIL))
			continue;
		for (j = i + 1; j < n; j++) {
			if (s[j] == symbol(NIL))
				continue;
			sfac_product_f(s, i, j);
		}
	}
	push(one);
	for (i = 0; i < n; i++) {
		if (s[i] == symbol(NIL))
			continue;
		push(s[i]);
		multiply();
	}
	p1 = pop();
	tos -= n;
	push(p1);
}

void
sfac_product_f(U **s, int a, int b)
{
	int i, n;
	p1 = s[a];
	p2 = s[b];
	if (ispower(p1)) {
		p3 = caddr(p1);
		p1 = cadr(p1);
	} else
		p3 = one;
	if (ispower(p2)) {
		p4 = caddr(p2);
		p2 = cadr(p2);
	} else
		p4 = one;
	if (isfactorial(p1) && isfactorial(p2)) {
		// Determine if the powers cancel.
		push(p3);
		push(p4);
		add();
		yyexpand();
		n = pop_integer();
		if (n != 0)
			return;
		// Find the difference between the two factorial args.
		// For example, the difference between (a + 2)! and a! is 2.
		push(cadr(p1));
		push(cadr(p2));
		subtract();
		yyexpand(); // to simplify
		n = pop_integer();
		if (n == 0 || n == (int) 0x80000000)
			return;
		if (n < 0) {
			n = -n;
			p5 = p1;
			p1 = p2;
			p2 = p5;
			p5 = p3;
			p3 = p4;
			p4 = p5;
		}
		push(one);
		for (i = 1; i <= n; i++) {
			push(cadr(p2));
			push_integer(i);
			add();
			push(p3);
			power();
			multiply();
		}
		s[a] = pop();
		s[b] = symbol(NIL);
	}
}

// Factor a polynomial

int expo;
U **polycoeff;

#undef POLY
#undef X
#undef Z
#undef A
#undef B
#undef Q
#undef RESULT
#undef YFACTOR

#define POLY p1
#define X p2
#define Z p3
#define A p4
#define B p5
#define Q p6
#define RESULT p7
#define YFACTOR p8

void
factorpoly(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (!find(p1, p2)) {
		push(p1);
		restore();
		return;
	}
	if (!ispoly(p1, p2)) {
		push(p1);
		restore();
		return;
	}
	if (!issymbol(p2)) {
		push(p1);
		restore();
		return;
	}
	push(p1);
	push(p2);
	yyfactorpoly();
	restore();
}

//-----------------------------------------------------------------------------
//
//	Input:		tos-2		true polynomial
//
//			tos-1		free variable
//
//	Output:		factored polynomial on stack
//
//-----------------------------------------------------------------------------

void
yyfactorpoly(void)
{
	int h, i;
	save();
	X = pop();
	POLY = pop();
	h = tos;
	if (isfloating(POLY))
		stop("floating point numbers in polynomial");
	polycoeff = stack + tos;
	push(POLY);
	push(X);
	expo = coeff() - 1;
	rationalize_coefficients(h);
	// for univariate polynomials we could do expo > 1
	while (expo > 0) {
		if (iszero(polycoeff[0])) {
			push_integer(1);
			A = pop();
			push_integer(0);
			B = pop();
		} else if (get_factor() == 0) {
			break;
		}
		push(A);
		push(X);
		multiply();
		push(B);
		add();
		YFACTOR = pop();
		// factor out negative sign (not req'd because A > 1)
#if 0
		if (isnegativeterm(A)) {
			push(FACTOR);
			negate();
			FACTOR = pop();
			push(RESULT);
			negate_noexpand();
			RESULT = pop();
		}
#endif
		push(RESULT);
		push(YFACTOR);
		multiply_noexpand();
		RESULT = pop();
		yydivpoly();
		while (expo && iszero(polycoeff[expo]))
			expo--;
	}
	// unfactored polynomial
	push(zero);
	for (i = 0; i <= expo; i++) {
		push(polycoeff[i]);
		push(X);
		push_integer(i);
		power();
		multiply();
		add();
	}
	POLY = pop();
	// factor out negative sign
	if (expo > 0 && isnegativeterm(polycoeff[expo])) {
		push(POLY);
		negate();
		POLY = pop();
		push(RESULT);
		negate_noexpand();
		RESULT = pop();
	}
	push(RESULT);
	push(POLY);
	multiply_noexpand();
	RESULT = pop();
	stack[h] = RESULT;
	tos = h + 1;
	restore();
}

void
rationalize_coefficients(int h)
{
	int i;
	// LCM of all polynomial coefficients
	RESULT = one;
	for (i = h; i < tos; i++) {
		push(stack[i]);
		denominator();
		push(RESULT);
		lcm();
		RESULT = pop();
	}
	// multiply each coefficient by RESULT
	for (i = h; i < tos; i++) {
		push(RESULT);
		push(stack[i]);
		multiply();
		stack[i] = pop();
	}
	// reciprocate RESULT
	push(RESULT);
	reciprocate();
	RESULT = pop();
}

int
get_factor(void)
{
	int i, j, h;
	int a0, an, na0, nan;
	h = tos;
	an = tos;
	push(polycoeff[expo]);
	divisors_onstack();
	nan = tos - an;
	a0 = tos;
	push(polycoeff[0]);
	divisors_onstack();
	na0 = tos - a0;
	// try roots
	for (i = 0; i < nan; i++) {
		for (j = 0; j < na0; j++) {
			A = stack[an + i];
			B = stack[a0 + j];
			push(B);
			push(A);
			divide();
			negate();
			Z = pop();
			evalpoly();
			if (iszero(Q)) {
				tos = h;
				return 1;
			}
			push(B);
			negate();
			B = pop();
			push(Z);
			negate();
			Z = pop();
			evalpoly();
			if (iszero(Q)) {
				tos = h;
				return 1;
			}
		}
	}
	tos = h;
	return 0;
}

//-----------------------------------------------------------------------------
//
//	Divide a polynomial by Ax+B
//
//	Input:		polycoeff	Dividend coefficients
//
//			expo		Degree of dividend
//
//			A		As above
//
//			B		As above
//
//	Output:		polycoeff	Contains quotient coefficients
//
//-----------------------------------------------------------------------------

void
yydivpoly(void)
{
	int i;
	Q = zero;
	for (i = expo; i > 0; i--) {
		push(polycoeff[i]);
		polycoeff[i] = Q;
		push(A);
		divide();
		Q = pop();
		push(polycoeff[i - 1]);
		push(Q);
		push(B);
		multiply();
		subtract();
		polycoeff[i - 1] = pop();
	}
	polycoeff[0] = Q;
}

void
evalpoly(void)
{
	int i;
	push(zero);
	for (i = expo; i >= 0; i--) {
		push(Z);
		multiply();
		push(polycoeff[i]);
		add();
	}
	Q = pop();
}

// Push expression factors onto the stack. For example...
//
// Input
//
//       2
//     3x  + 2x + 1
//
// Output on stack
//
//     [  3  ]
//     [ x^2 ]
//     [  2  ]
//     [  x  ]
//     [  1  ]
//
// but not necessarily in that order. Returns the number of factors.

// Local U *p is OK here because no functional path to garbage collector.

int
factors(U *p)
{
	int h = tos;
	if (car(p) == symbol(ADD)) {
		p = cdr(p);
		while (iscons(p)) {
			push_term_factors(car(p));
			p = cdr(p);
		}
	} else
		push_term_factors(p);
	return tos - h;
}

// Local U *p is OK here because no functional path to garbage collector.

void
push_term_factors(U *p)
{
	if (car(p) == symbol(MULTIPLY)) {
		p = cdr(p);
		while (iscons(p)) {
			push(car(p));
			p = cdr(p);
		}
	} else
		push(p);
}

/* Remove terms that involve a given symbol or expression. For example...

	filter(x^2 + x + 1, x)		=>	1

	filter(x^2 + x + 1, x^2)	=>	x + 1
*/

void
eval_filter(void)
{
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		filter();
		p1 = cdr(p1);
	}
}

/* For example...

	push(F)
	push(X)
	filter()
	F = pop()
*/

void
filter(void)
{
	save();
	p2 = pop();
	p1 = pop();
	filter_main();
	restore();
}

void
filter_main(void)
{
	if (car(p1) == symbol(ADD))
		filter_sum();
	else if (istensor(p1))
		filter_tensor();
	else if (find(p1, p2))
		push_integer(0);
	else
		push(p1);
}

void
filter_sum(void)
{
	push_integer(0);
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		push(p2);
		filter();
		add();
		p1 = cdr(p1);
	}
}

void
filter_tensor(void)
{
	int i, n;
	n = p1->u.tensor->nelem;
	p3 = alloc_tensor(n);
	p3->u.tensor->ndim = p1->u.tensor->ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	for (i = 0; i < n; i++) {
		push(p1->u.tensor->elem[i]);
		push(p2);
		filter();
		p3->u.tensor->elem[i] = pop();
	}
	push(p3);
}

// returns 1 if expr p contains expr q, otherweise returns 0

int
find(U *p, U *q)
{
	int i;
	if (equal(p, q))
		return 1;
	if (istensor(p)) {
		for (i = 0; i < p->u.tensor->nelem; i++)
			if (find(p->u.tensor->elem[i], q))
				return 1;
		return 0;
	}
	while (iscons(p)) {
		if (find(car(p), q))
			return 1;
		p = cdr(p);
	}
	return 0;
}

void
eval_float(void)
{
	push(cadr(p1));
	eval();
	yyfloat();
	eval(); // normalize
}

void
yyfloat(void)
{
	int i, h;
	save();
	p1 = pop();
	if (iscons(p1)) {
		h = tos;
		while (iscons(p1)) {
			push(car(p1));
			yyfloat();
			p1 = cdr(p1);
		}
		list(tos - h);
	} else if (p1->k == TENSOR) {
		push(p1);
		copy_tensor();
		p1 = pop();
		for (i = 0; i < p1->u.tensor->nelem; i++) {
			push(p1->u.tensor->elem[i]);
			yyfloat();
			p1->u.tensor->elem[i] = pop();
		}
		push(p1);
	} else if (p1->k == NUM) {
		push(p1);
		bignum_float();
	} else if (p1 == symbol(PI))
		push_double(M_PI);
	else if (p1 == symbol(EXP1))
		push_double(M_E);
	else
		push(p1);
	restore();
}

void
eval_floor(void)
{
	push(cadr(p1));
	eval();
	yfloor();
}

void
yfloor(void)
{
	save();
	yyfloor();
	restore();
}

void
yyfloor(void)
{
	double d;
	p1 = pop();
	if (!isnum(p1)) {
		push_symbol(FLOOR);
		push(p1);
		list(2);
		return;
	}
	if (isdouble(p1)) {
		d = floor(p1->u.d);
		push_double(d);
		return;
	}
	if (isinteger(p1)) {
		push(p1);
		return;
	}
	p3 = alloc();
	p3->k = NUM;
	p3->u.q.a = mdiv(p1->u.q.a, p1->u.q.b);
	p3->u.q.b = mint(1);
	push(p3);
	if (isnegativenumber(p1)) {
		push_integer(-1);
		add();
	}
}

// 'for' function

#undef I
#undef X

#define I p5
#define X p6

void
eval_for(void)
{
	int i, j, k;
	// 1st arg (quoted)
	X = cadr(p1);
	if (!issymbol(X))
		stop("for: 1st arg?");
	// 2nd arg
	push(caddr(p1));
	eval();
	j = pop_integer();
	if (j == (int) 0x80000000)
		stop("for: 2nd arg?");
	// 3rd arg
	push(cadddr(p1));
	eval();
	k = pop_integer();
	if (k == (int) 0x80000000)
		stop("for: 3rd arg?");
	// remaining args
	p1 = cddddr(p1);
	push_binding(X);
	for (i = j; i <= k; i++) {
		push_integer(i);
		I = pop();
		set_binding(X, I);
		p2 = p1;
		while (iscons(p2)) {
			push(car(p2));
			eval();
			pop();
			p2 = cdr(p2);
		}
	}
	pop_binding(X);
	// return value
	push_symbol(NIL);
}

// Greatest common denominator

void
eval_gcd(void)
{
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		gcd();
		p1 = cdr(p1);
	}
}

void
gcd(void)
{
	int x = expanding;
	save();
	gcd_main();
	restore();
	expanding = x;
}

void
gcd_main(void)
{
	expanding = 1;
	p2 = pop();
	p1 = pop();
	if (equal(p1, p2)) {
		push(p1);
		return;
	}
	if (isrational(p1) && isrational(p2)) {
		push(p1);
		push(p2);
		gcd_numbers();
		return;
	}
	if (car(p1) == symbol(ADD) && car(p2) == symbol(ADD)) {
		gcd_expr_expr();
		return;
	}
	if (car(p1) == symbol(ADD)) {
		gcd_expr(p1);
		p1 = pop();
	}
	if (car(p2) == symbol(ADD)) {
		gcd_expr(p2);
		p2 = pop();
	}
	if (car(p1) == symbol(MULTIPLY) && car(p2) == symbol(MULTIPLY)) {
		gcd_term_term();
		return;
	}
	if (car(p1) == symbol(MULTIPLY)) {
		gcd_term_factor();
		return;
	}
	if (car(p2) == symbol(MULTIPLY)) {
		gcd_factor_term();
		return;
	}
	// gcd of factors
	if (car(p1) == symbol(POWER)) {
		p3 = caddr(p1);
		p1 = cadr(p1);
	} else
		p3 = one;
	if (car(p2) == symbol(POWER)) {
		p4 = caddr(p2);
		p2 = cadr(p2);
	} else
		p4 = one;
	if (!equal(p1, p2)) {
		push(one);
		return;
	}
	// are both exponents numerical?
	if (isnum(p3) && isnum(p4)) {
		push(p1);
		if (lessp(p3, p4))
			push(p3);
		else
			push(p4);
		power();
		return;
	}
	// are the exponents multiples of eah other?
	push(p3);
	push(p4);
	divide();
	p5 = pop();
	if (isnum(p5)) {
		push(p1);
		// choose the smallest exponent
		if (car(p3) == symbol(MULTIPLY) && isnum(cadr(p3)))
			p5 = cadr(p3);
		else
			p5 = one;
		if (car(p4) == symbol(MULTIPLY) && isnum(cadr(p4)))
			p6 = cadr(p4);
		else
			p6 = one;
		if (lessp(p5, p6))
			push(p3);
		else
			push(p4);
		power();
		return;
	}
	push(p3);
	push(p4);
	subtract();
	p5 = pop();
	if (!isnum(p5)) {
		push(one);
		return;
	}
	// can't be equal because of test near beginning
	push(p1);
	if (isnegativenumber(p5))
		push(p3);
	else
		push(p4);
	power();
}

// in this case gcd is used as a composite function, i.e. gcd(gcd(gcd...

void
gcd_expr_expr(void)
{
	if (length(p1) != length(p2)) {
		push(one);
		return;
	}
	p3 = cdr(p1);
	push(car(p3));
	p3 = cdr(p3);
	while (iscons(p3)) {
		push(car(p3));
		gcd();
		p3 = cdr(p3);
	}
	p3 = pop();
	p4 = cdr(p2);
	push(car(p4));
	p4 = cdr(p4);
	while (iscons(p4)) {
		push(car(p4));
		gcd();
		p4 = cdr(p4);
	}
	p4 = pop();
	push(p1);
	push(p3);
	divide();
	p5 = pop();
	push(p2);
	push(p4);
	divide();
	p6 = pop();
	if (equal(p5, p6)) {
		push(p5);
		push(p3);
		push(p4);
		gcd();
		multiply();
	} else
		push(one);
}

void
gcd_expr(U *p)
{
	p = cdr(p);
	push(car(p));
	p = cdr(p);
	while (iscons(p)) {
		push(car(p));
		gcd();
		p = cdr(p);
	}
}

void
gcd_term_term(void)
{
	push(one);
	p3 = cdr(p1);
	while (iscons(p3)) {
		p4 = cdr(p2);
		while (iscons(p4)) {
			push(car(p3));
			push(car(p4));
			gcd();
			multiply();
			p4 = cdr(p4);
		}
		p3 = cdr(p3);
	}
}

void
gcd_term_factor(void)
{
	push(one);
	p3 = cdr(p1);
	while (iscons(p3)) {
		push(car(p3));
		push(p2);
		gcd();
		multiply();
		p3 = cdr(p3);
	}
}

void
gcd_factor_term(void)
{
	push(one);
	p4 = cdr(p2);
	while (iscons(p4)) {
		push(p1);
		push(car(p4));
		gcd();
		multiply();
		p4 = cdr(p4);
	}
}

// Guess which symbol to use for derivative, integral, etc.

void
guess(void)
{
	U *p;
	p = pop();
	push(p);
	if (find(p, symbol(SYMBOL_X)))
		push_symbol(SYMBOL_X);
	else if (find(p, symbol(SYMBOL_Y)))
		push_symbol(SYMBOL_Y);
	else if (find(p, symbol(SYMBOL_Z)))
		push_symbol(SYMBOL_Z);
	else if (find(p, symbol(SYMBOL_T)))
		push_symbol(SYMBOL_T);
	else if (find(p, symbol(SYMBOL_S)))
		push_symbol(SYMBOL_S);
	else
		push_symbol(SYMBOL_X);
}

//-----------------------------------------------------------------------------
//
//	Hermite polynomial
//
//	Input:		tos-2		x	(can be a symbol or expr)
//
//			tos-1		n
//
//	Output:		Result on stack
//
//-----------------------------------------------------------------------------

void
hermite(void)
{
	save();
	yyhermite();
	restore();
}

// uses the recurrence relation H(x,n+1)=2*x*H(x,n)-2*n*H(x,n-1)

#undef X
#undef N
#undef Y
#undef Y1
#undef Y0

#define X p1
#define N p2
#define Y p3
#define Y1 p4
#define Y0 p5

void
yyhermite(void)
{
	int n;
	N = pop();
	X = pop();
	push(N);
	n = pop_integer();
	if (n < 0) {
		push_symbol(HERMITE);
		push(X);
		push(N);
		list(3);
		return;
	}
	if (issymbol(X))
		yyhermite2(n);
	else {
		Y = X;			// do this when X is an expr
		X = symbol(SPECX);
		yyhermite2(n);
		X = Y;
		push_symbol(SPECX);
		push(X);
		subst();
		eval();
	}
}

void
yyhermite2(int n)
{
	int i;
	push_integer(1);
	push_integer(0);
	Y1 = pop();
	for (i = 0; i < n; i++) {
		Y0 = Y1;
		Y1 = pop();
		push(X);
		push(Y1);
		multiply();
		push_integer(i);
		push(Y0);
		multiply();
		subtract();
		push_integer(2);
		multiply();
	}
}

//-----------------------------------------------------------------------------
//
//	Create a Hilbert matrix
//
//	Input:		Dimension on stack
//
//	Output:		Hilbert matrix on stack
//
//	Example:
//
//	> hilbert(5)
//	((1,1/2,1/3,1/4),(1/2,1/3,1/4,1/5),(1/3,1/4,1/5,1/6),(1/4,1/5,1/6,1/7))
//
//-----------------------------------------------------------------------------

#undef A
#undef N
#undef AELEM

#define A p1
#define N p2
#define AELEM(i, j) A->u.tensor->elem[i * n + j]

void
hilbert(void)
{
	int i, j, n;
	save();
	N = pop();
	push(N);
	n = pop_integer();
	if (n < 2) {
		push_symbol(HILBERT);
		push(N);
		list(2);
		restore();
		return;
	}
	push_zero_matrix(n, n);
	A = pop();
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			push_integer(i + j + 1);
			inverse();
			AELEM(i, j) = pop();
		}
	}
	push(A);
	restore();
}

/* Returns the coefficient of the imaginary part of complex z

	z		imag(z)
	-		-------

	a + i b		b

	exp(i a)	sin(a)
*/

void
eval_imag(void)
{
	push(cadr(p1));
	eval();
	imag();
}

void
imag(void)
{
	save();
	rect();
	p1 = pop();
	push(p1);
	push(p1);
	conjugate();
	subtract();
	push_integer(2);
	divide();
	push(imaginaryunit);
	divide();
	restore();
}

// n is the total number of things on the stack. The first thing on the stack
// is the object to be indexed, followed by the indices themselves.

void
index_function(int n)
{
	int i, k, m, ndim, nelem, t;
	U **s;
	save();
	s = stack + tos - n;
	p1 = s[0];
	// index of number (FIXME include complex numbers)
	if (isnum(p1)) {
		tos -= n;
		push(p1);
		restore();
		return;
	}
	// index of symbol (f.e., u[2] -> u[2])
	if (!istensor(p1)) {
		list(n);
		p1 = pop();
		push(symbol(INDEX));
		push(p1);
		append();
		restore();
		return;
	}
	ndim = p1->u.tensor->ndim;
	m = n - 1;
	if (m > ndim)
		stop("too many indices for tensor");
	k = 0;
	for (i = 0; i < m; i++) {
		push(s[i + 1]);
		t = pop_integer();
		if (t < 1 || t > p1->u.tensor->dim[i])
			stop("index out of range");
		k = k * p1->u.tensor->dim[i] + t - 1;
	}
	if (ndim == m) {
		tos -= n;
		push(p1->u.tensor->elem[k]);
		restore();
		return;
	}
	for (i = m; i < ndim; i++)
		k = k * p1->u.tensor->dim[i] + 0;
	nelem = 1;
	for (i = m; i < ndim; i++)
		nelem *= p1->u.tensor->dim[i];
	p2 = alloc_tensor(nelem);
	p2->u.tensor->ndim = ndim - m;
	for (i = m; i < ndim; i++)
		p2->u.tensor->dim[i - m] = p1->u.tensor->dim[i];
	for (i = 0; i < nelem; i++)
		p2->u.tensor->elem[i] = p1->u.tensor->elem[k + i];
	tos -= n;
	push(p2);
	restore();
}

//-----------------------------------------------------------------------------
//
//	Input:		n		Number of args on stack
//
//			tos-n		Right-hand value
//
//			tos-n+1		Left-hand value
//
//			tos-n+2		First index
//
//			.
//			.
//			.
//
//			tos-1		Last index
//
//	Output:		Result on stack
//
//-----------------------------------------------------------------------------

#define LVALUE p1
#define RVALUE p2
#undef TMP
#define TMP p3

void
set_component(int n)
{
	int i, k, m, ndim, t;
	U **s;
	save();
	if (n < 3)
		stop("error in indexed assign");
	s = stack + tos - n;
	RVALUE = s[0];
	LVALUE = s[1];
	if (!istensor(LVALUE))
		stop("error in indexed assign");
	ndim = LVALUE->u.tensor->ndim;
	m = n - 2;
	if (m > ndim)
		stop("error in indexed assign");
	k = 0;
	for (i = 0; i < m; i++) {
		push(s[i + 2]);
		t = pop_integer();
		if (t < 1 || t > LVALUE->u.tensor->dim[i])
			stop("error in indexed assign\n");
		k = k * p1->u.tensor->dim[i] + t - 1;
	}
	for (i = m; i < ndim; i++)
		k = k * p1->u.tensor->dim[i] + 0;
	// copy
	TMP = alloc_tensor(LVALUE->u.tensor->nelem);
	TMP->u.tensor->ndim = LVALUE->u.tensor->ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		TMP->u.tensor->dim[i] = LVALUE->u.tensor->dim[i];
	for (i = 0; i < p1->u.tensor->nelem; i++)
		TMP->u.tensor->elem[i] = LVALUE->u.tensor->elem[i];
	LVALUE = TMP;
	if (ndim == m) {
		if (istensor(RVALUE))
			stop("error in indexed assign");
		LVALUE->u.tensor->elem[k] = RVALUE;
		tos -= n;
		push(LVALUE);
		restore();
		return;
	}
	// see if the rvalue matches
	if (!istensor(RVALUE))
		stop("error in indexed assign");
	if (ndim - m != RVALUE->u.tensor->ndim)
		stop("error in indexed assign");
	for (i = 0; i < RVALUE->u.tensor->ndim; i++)
		if (LVALUE->u.tensor->dim[m + i] != RVALUE->u.tensor->dim[i])
			stop("error in indexed assign");
	// copy rvalue
	for (i = 0; i < RVALUE->u.tensor->nelem; i++)
		LVALUE->u.tensor->elem[k + i] = RVALUE->u.tensor->elem[i];
	tos -= n;
	push(LVALUE);
	restore();
}

void
init(void)
{
	init_symbol_table();
	tos = 0;
	esc_flag = 0;
	draw_flag = 0;
	frame = stack + TOS;
	p0 = symbol(NIL);
	p1 = symbol(NIL);
	p2 = symbol(NIL);
	p3 = symbol(NIL);
	p4 = symbol(NIL);
	p5 = symbol(NIL);
	p6 = symbol(NIL);
	p7 = symbol(NIL);
	p8 = symbol(NIL);
	p9 = symbol(NIL);
	// caution, jmp_buf not set up, stop() will crash
	push_integer(0);
	binding[V0] = pop();
	push_integer(1);
	binding[V1] = pop();
	push_symbol(POWER);
	push_integer(-1);
	push_rational(1, 2);
	list(3);
	binding[IU] = pop();
}

void
init_symbol_table(void)
{
	int i;
	for (i = 0; i < NSYM; i++) {
		symtab[i].k = SYM;
		binding[i] = symtab + i;
		arglist[i] = symbol(NIL);
	}
	std_symbol("abs", ABS);
	std_symbol("add", ADD);
	std_symbol("adj", ADJ);
	std_symbol("and", AND);
	std_symbol("arccos", ARCCOS);
	std_symbol("arccosh", ARCCOSH);
	std_symbol("arcsin", ARCSIN);
	std_symbol("arcsinh", ARCSINH);
	std_symbol("arctan", ARCTAN);
	std_symbol("arctanh", ARCTANH);
	std_symbol("arg", ARG);
	std_symbol("atomize", ATOMIZE);
	std_symbol("besselj", BESSELJ);
	std_symbol("bessely", BESSELY);
	std_symbol("binding", BINDING);
	std_symbol("binomial", BINOMIAL);
	std_symbol("ceiling", CEILING);
	std_symbol("check", CHECK);
	std_symbol("choose", CHOOSE);
	std_symbol("circexp", CIRCEXP);
	std_symbol("clock", CLOCK);
	std_symbol("coeff", COEFF);
	std_symbol("cofactor", COFACTOR);
	std_symbol("condense", CONDENSE);
	std_symbol("conj", CONJ);
	std_symbol("contract", CONTRACT);
	std_symbol("cos", COS);
	std_symbol("cosh", COSH);
	std_symbol("decomp", DECOMP);
	std_symbol("defint", DEFINT);
	std_symbol("deg", DEGREE);
	std_symbol("denominator", DENOMINATOR);
	std_symbol("det", DET);
	std_symbol("derivative", DERIVATIVE);
	std_symbol("dim", DIM);
	std_symbol("display", DISPLAY);
	std_symbol("divisors", DIVISORS);
	std_symbol("do", DO);
	std_symbol("dot", DOT);
	std_symbol("draw", DRAW);
	std_symbol("erf", ERF);
	std_symbol("erfc", ERFC);
	std_symbol("eigen", EIGEN);
	std_symbol("eigenval", EIGENVAL);
	std_symbol("eigenvec", EIGENVEC);
	std_symbol("eval", EVAL);
	std_symbol("exp", EXP);
	std_symbol("expand", EXPAND);
	std_symbol("expcos", EXPCOS);
	std_symbol("expsin", EXPSIN);
	std_symbol("factor", FACTOR);
	std_symbol("factorial", FACTORIAL);
	std_symbol("factorpoly", FACTORPOLY);
	std_symbol("filter", FILTER);
	std_symbol("float", FLOATF);
	std_symbol("floor", FLOOR);
	std_symbol("for", FOR);
	std_symbol("gcd", GCD);
	std_symbol("hermite", HERMITE);
	std_symbol("hilbert", HILBERT);
	std_symbol("imag", IMAG);
	std_symbol("component", INDEX);
	std_symbol("inner", INNER);
	std_symbol("integral", INTEGRAL);
	std_symbol("inv", INV);
	std_symbol("invg", INVG);
	std_symbol("isinteger", ISINTEGER);
	std_symbol("isprime", ISPRIME);
	std_symbol("laguerre", LAGUERRE);
	std_symbol("lcm", LCM);
	std_symbol("leading", LEADING);
	std_symbol("legendre", LEGENDRE);
	std_symbol("log", LOG);
	std_symbol("mag", MAG);
	std_symbol("mod", MOD);
	std_symbol("multiply", MULTIPLY);
	std_symbol("not", NOT);
	std_symbol("nroots", NROOTS);
	std_symbol("number", NUMBER);
	std_symbol("numerator", NUMERATOR);
	std_symbol("operator", OPERATOR);
	std_symbol("or", OR);
	std_symbol("outer", OUTER);
	std_symbol("polar", POLAR);
	std_symbol("power", POWER);
	std_symbol("prime", PRIME);
	std_symbol("print", PRINT);
	std_symbol("product", PRODUCT);
	std_symbol("quote", QUOTE);
	std_symbol("quotient", QUOTIENT);
	std_symbol("rank", RANK);
	std_symbol("rationalize", RATIONALIZE);
	std_symbol("real", REAL);
	std_symbol("rect", YYRECT);
	std_symbol("roots", ROOTS);
	std_symbol("equals", SETQ);
	std_symbol("sgn", SGN);
	std_symbol("simplify", SIMPLIFY);
	std_symbol("sin", SIN);
	std_symbol("sinh", SINH);
	std_symbol("sqrt", SQRT);
	std_symbol("stop", STOP);
	std_symbol("subst", SUBST);
	std_symbol("sum", SUM);
	std_symbol("tan", TAN);
	std_symbol("tanh", TANH);
	std_symbol("taylor", TAYLOR);
	std_symbol("test", TEST);
	std_symbol("testeq", TESTEQ);
	std_symbol("testge", TESTGE);
	std_symbol("testgt", TESTGT);
	std_symbol("testle", TESTLE);
	std_symbol("testlt", TESTLT);
	std_symbol("transpose", TRANSPOSE);
	std_symbol("unit", UNIT);
	std_symbol("zero", ZERO);
	std_symbol("$", MARK1);
	std_symbol("~", NATNUM); // tilde so sort after other symbols
	std_symbol("nil", NIL);
	std_symbol("pi", PI);
	std_symbol("$0", V0);
	std_symbol("$1", V1);
	std_symbol("$I", IU);
	std_symbol("$", MARK2);
	std_symbol("$a", METAA); // must be distinct so they sort correctly
	std_symbol("$b", METAB);
	std_symbol("$x", METAX);
	std_symbol("$X", SPECX);
	std_symbol("a", SYMBOL_A);
	std_symbol("b", SYMBOL_B);
	std_symbol("c", SYMBOL_C);
	std_symbol("d", SYMBOL_D);
	std_symbol("i", SYMBOL_I);
	std_symbol("j", SYMBOL_J);
	std_symbol("n", SYMBOL_N);
	std_symbol("r", SYMBOL_R);
	std_symbol("s", SYMBOL_S);
	std_symbol("t", SYMBOL_T);
	std_symbol("x", SYMBOL_X);
	std_symbol("y", SYMBOL_Y);
	std_symbol("z", SYMBOL_Z);
	std_symbol("autoexpand", AUTOEXPAND);
	std_symbol("bake", BAKE);
	std_symbol("last", LAST);
	std_symbol("trace", TRACE);
	std_symbol("tty", TTY);
	std_symbol("$", MARK3);
}

// Do the inner product of tensors.

void
eval_inner(void)
{
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		inner();
		p1 = cdr(p1);
	}
}

void
inner(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (istensor(p1) && istensor(p2))
		inner_f();
	else {
		push(p1);
		push(p2);
		if (istensor(p1))
			tensor_times_scalar();
		else if (istensor(p2))
			scalar_times_tensor();
		else
			multiply();
	}
	restore();
}

// inner product of tensors p1 and p2

void
inner_f(void)
{
	int ak, bk, i, j, k, n, ndim;
	U **a, **b, **c;
	n = p1->u.tensor->dim[p1->u.tensor->ndim - 1];
	if (n != p2->u.tensor->dim[0])
		stop("inner: tensor dimension check");
	ndim = p1->u.tensor->ndim + p2->u.tensor->ndim - 2;
	if (ndim > MAXDIM)
		stop("inner: rank of result exceeds maximum");
	a = p1->u.tensor->elem;
	b = p2->u.tensor->elem;
	//---------------------------------------------------------------------
	//
	//	ak is the number of rows in tensor A
	//
	//	bk is the number of columns in tensor B
	//
	//	Example:
	//
	//	A[3][3][4] B[4][4][3]
	//
	//	  3  3				ak = 3 * 3 = 9
	//
	//	                4  3		bk = 4 * 3 = 12
	//
	//---------------------------------------------------------------------
	ak = 1;
	for (i = 0; i < p1->u.tensor->ndim - 1; i++)
		ak *= p1->u.tensor->dim[i];
	bk = 1;
	for (i = 1; i < p2->u.tensor->ndim; i++)
		bk *= p2->u.tensor->dim[i];
	p3 = alloc_tensor(ak * bk);
	c = p3->u.tensor->elem;
	for (i = 0; i < ak; i++) {
		for (j = 0; j < n; j++) {
			if (iszero(a[i * n + j]))
				continue;
			for (k = 0; k < bk; k++) {
				push(a[i * n + j]);
				push(b[j * bk + k]);
				multiply();
				push(c[i * bk + k]);
				add();
				c[i * bk + k] = pop();
			}
		}
	}
	//---------------------------------------------------------------------
	//
	//	Note on understanding "k * bk + j"
	//
	//	k * bk because each element of a column is bk locations apart
	//
	//	+ j because the beginnings of all columns are in the first bk
	//	locations
	//
	//	Example: n = 2, bk = 6
	//
	//	b111	<- 1st element of 1st column
	//	b112	<- 1st element of 2nd column
	//	b113	<- 1st element of 3rd column
	//	b121	<- 1st element of 4th column
	//	b122	<- 1st element of 5th column
	//	b123	<- 1st element of 6th column
	//
	//	b211	<- 2nd element of 1st column
	//	b212	<- 2nd element of 2nd column
	//	b213	<- 2nd element of 3rd column
	//	b221	<- 2nd element of 4th column
	//	b222	<- 2nd element of 5th column
	//	b223	<- 2nd element of 6th column
	//
	//---------------------------------------------------------------------
	if (ndim == 0)
		push(p3->u.tensor->elem[0]);
	else {
		p3->u.tensor->ndim = ndim;
		for (i = 0; i < p1->u.tensor->ndim - 1; i++)
			p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
		j = i;
		for (i = 0; i < p2->u.tensor->ndim - 1; i++)
			p3->u.tensor->dim[j + i] = p2->u.tensor->dim[i + 1];
		push(p3);
	}
}

#undef X

#define X p3

void
eval_integral(void)
{
	int i, n;
	// evaluate 1st arg to get function F
	p1 = cdr(p1);
	push(car(p1));
	eval();
	// check for single arg
	if (cdr(p1) == symbol(NIL)) {
		guess();
		integral();
		return;
	}
	p1 = cdr(p1);
	while (iscons(p1)) {
		// next arg should be a symbol
		push(car(p1)); // have to eval in case of $METAX
		eval();
		X = pop();
		if (!issymbol(X))
			stop("integral: symbol expected");
		p1 = cdr(p1);
		// if next arg is a number then use it
		n = 1;
		if (isnum(car(p1))) {
			push(car(p1));
			n = pop_integer();
			if (n < 1)
				stop("nth integral: check n");
			p1 = cdr(p1);
		}
		for (i = 0; i < n; i++) {
			push(X);
			integral();
		}
	}
}

void
integral(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (car(p1) == symbol(ADD))
		integral_of_sum();
	else if (car(p1) == symbol(MULTIPLY))
		integral_of_product();
	else
		integral_of_form();
	p1 = pop();
	if (find(p1, symbol(INTEGRAL)))
		stop("integral: sorry, could not find a solution");
	push(p1);
	simplify();	// polish the result
	eval();		// normalize the result
	restore();
}

void
integral_of_sum(void)
{
	p1 = cdr(p1);
	push(car(p1));
	push(p2);
	integral();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		push(p2);
		integral();
		add();
		p1 = cdr(p1);
	}
}

void
integral_of_product(void)
{
	push(p1);
	push(p2);
	partition();
	p1 = pop();			// pop variable part
	integral_of_form();
	multiply();			// multiply constant part
}

extern char *itab[];

void
integral_of_form(void)
{
	push(p1);
	push(p2);
	transform(itab);
	p3 = pop();
	if (p3 == symbol(NIL)) {
		push_symbol(INTEGRAL);
		push(p1);
		push(p2);
		list(3);
	} else
		push(p3);
}

//-----------------------------------------------------------------------------
//
//	Input:		Matrix on stack
//
//	Output:		Inverse on stack
//
//	Example:
//
//	> inv(((1,2),(3,4))
//	((-2,1),(3/2,-1/2))
//
//	Note:
//
//	Uses Gaussian elimination for numerical matrices.
//
//-----------------------------------------------------------------------------

int
inv_check_arg(void)
{
	if (!istensor(p1))
		return 0;
	else if (p1->u.tensor->ndim != 2)
		return 0;
	else if (p1->u.tensor->dim[0] != p1->u.tensor->dim[1])
		return 0;
	else
		return 1;
}

void
inv(void)
{
	int i, n;
	U **a;
	save();
	p1 = pop();
	if (inv_check_arg() == 0) {
		push_symbol(INV);
		push(p1);
		list(2);
		restore();
		return;
	}
	n = p1->u.tensor->nelem;
	a = p1->u.tensor->elem;
	for (i = 0; i < n; i++)
		if (!isnum(a[i]))
			break;
	if (i == n)
		yyinvg();
	else {
		push(p1);
		adj();
		push(p1);
		det();
		p2 = pop();
		if (iszero(p2))
			stop("inverse of singular matrix");
		push(p2);
		divide();
	}
	restore();
}

void
invg(void)
{
	save();
	p1 = pop();
	if (inv_check_arg() == 0) {
		push_symbol(INVG);
		push(p1);
		list(2);
		restore();
		return;
	}
	yyinvg();
	restore();
}

// inverse using gaussian elimination

void
yyinvg(void)
{
	int h, i, j, n;
	n = p1->u.tensor->dim[0];
	h = tos;
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			if (i == j)
				push(one);
			else
				push(zero);
	for (i = 0; i < n * n; i++)
		push(p1->u.tensor->elem[i]);
	inv_decomp(n);
	p1 = alloc_tensor(n * n);
	p1->u.tensor->ndim = 2;
	p1->u.tensor->dim[0] = n;
	p1->u.tensor->dim[1] = n;
	for (i = 0; i < n * n; i++)
		p1->u.tensor->elem[i] = stack[h + i];
	tos -= 2 * n * n;
	push(p1);
}

//-----------------------------------------------------------------------------
//
//	Input:		n * n unit matrix on stack
//
//			n * n operand on stack
//
//	Output:		n * n inverse matrix on stack
//
//			n * n garbage on stack
//
//			p2 mangled
//
//-----------------------------------------------------------------------------

#undef A
#undef U

#define A(i, j) stack[a + n * (i) + (j)]
#define U(i, j) stack[u + n * (i) + (j)]

void
inv_decomp(int n)
{
	int a, d, i, j, u;
	a = tos - n * n;
	u = a - n * n;
	for (d = 0; d < n; d++) {
		// diagonal element zero?
		if (equal(A(d, d), zero)) {
			// find a new row
			for (i = d + 1; i < n; i++)
				if (!equal(A(i, d), zero))
					break;
			if (i == n)
				stop("inverse of singular matrix");
			// exchange rows
			for (j = 0; j < n; j++) {
				p2 = A(d, j);
				A(d, j) = A(i, j);
				A(i, j) = p2;
				p2 = U(d, j);
				U(d, j) = U(i, j);
				U(i, j) = p2;
			}
		}
		// multiply the pivot row by 1 / pivot
		p2 = A(d, d);
		for (j = 0; j < n; j++) {
			if (j > d) {
				push(A(d, j));
				push(p2);
				divide();
				A(d, j) = pop();
			}
			push(U(d, j));
			push(p2);
			divide();
			U(d, j) = pop();
		}
		// clear out the column above and below the pivot
		for (i = 0; i < n; i++) {
			if (i == d)
				continue;
			// multiplier
			p2 = A(i, d);
			// add pivot row to i-th row
			for (j = 0; j < n; j++) {
				if (j > d) {
					push(A(i, j));
					push(A(d, j));
					push(p2);
					multiply();
					subtract();
					A(i, j) = pop();
				}
				push(U(i, j));
				push(U(d, j));
				push(p2);
				multiply();
				subtract();
				U(i, j) = pop();
			}
		}
	}
}

int
iszero(U *p)
{
	int i;
	switch (p->k) {
	case NUM:
		if (MZERO(p->u.q.a))
			return 1;
		break;
	case DOUBLE:
		if (p->u.d == 0.0)
			return 1;
		break;
	case TENSOR:
		for (i = 0; i < p->u.tensor->nelem; i++)
			if (!iszero(p->u.tensor->elem[i]))
				return 0;
		return 1;
	default:
		break;
	}
	return 0;
}

int
isnegativenumber(U *p)
{
	switch (p->k) {
	case NUM:
		if (MSIGN(p->u.q.a) == -1)
			return 1;
		break;
	case DOUBLE:
		if (p->u.d < 0.0)
			return 1;
		break;
	default:
		break;
	}
	return 0;
}

int
isplusone(U *p)
{
	switch (p->k) {
	case NUM:
		if (MEQUAL(p->u.q.a, 1) && MEQUAL(p->u.q.b, 1))
			return 1;
		break;
	case DOUBLE:
		if (p->u.d == 1.0)
			return 1;
		break;
	default:
		break;
	}
	return 0;
}

int
isminusone(U *p)
{
	switch (p->k) {
	case NUM:
		if (MEQUAL(p->u.q.a, -1) && MEQUAL(p->u.q.b, 1))
			return 1;
		break;
	case DOUBLE:
		if (p->u.d == -1.0)
			return 1;
		break;
	default:
		break;
	}
	return 0;
}

int
isinteger(U *p)
{
	if (p->k == NUM && MEQUAL(p->u.q.b, 1))
		return 1;
	else
		return 0;
}

int
isnonnegativeinteger(U *p)
{
	if (isrational(p) && MEQUAL(p->u.q.b, 1) && MSIGN(p->u.q.a) == 1)
		return 1;
	else
		return 0;
}

int
isposint(U *p)
{
	if (isinteger(p) && MSIGN(p->u.q.a) == 1)
		return 1;
	else
		return 0;
}

int
ispoly(U *p, U *x)
{
	if (find(p, x))
		return ispoly_expr(p, x);
	else
		return 0;
}

int
ispoly_expr(U *p, U *x)
{
	if (car(p) == symbol(ADD)) {
		p = cdr(p);
		while (iscons(p)) {
			if (!ispoly_term(car(p), x))
				return 0;
			p = cdr(p);
		}
		return 1;
	} else
		return ispoly_term(p, x);
}

int
ispoly_term(U *p, U *x)
{
	if (car(p) == symbol(MULTIPLY)) {
		p = cdr(p);
		while (iscons(p)) {
			if (!ispoly_factor(car(p), x))
				return 0;
			p = cdr(p);
		}
		return 1;
	} else
		return ispoly_factor(p, x);
}

int
ispoly_factor(U *p, U *x)
{
	if (equal(p, x))
		return 1;
	if (car(p) == symbol(POWER) && equal(cadr(p), x)) {
		if (isposint(caddr(p)))
			return 1;
		else
			return 0;
	}
	if (find(p, x))
		return 0;
	else
		return 1;
}

int
isnegativeterm(U *p)
{
	if (isnegativenumber(p))
		return 1;
	else if (car(p) == symbol(MULTIPLY) && isnegativenumber(cadr(p)))
		return 1;
	else
		return 0;
}

int
isimaginarynumber(U *p)
{
	if ((car(p) == symbol(MULTIPLY)
	&& length(p) == 3
	&& isnum(cadr(p))
	&& equal(caddr(p), imaginaryunit))
	|| equal(p, imaginaryunit))
		return 1;
	else
		return 0;
}

int
iscomplexnumber(U *p)
{
	if ((car(p) == symbol(ADD)
	&& length(p) == 3
	&& isnum(cadr(p))
	&& isimaginarynumber(caddr(p)))
	|| isimaginarynumber(p))
		return 1;
	else
		return 0;
}

int
iseveninteger(U *p)
{
	if (isinteger(p) && (p->u.q.a[0] & 1) == 0)
		return 1;
	else
		return 0;
}

int
isnegative(U *p)
{
	if (car(p) == symbol(ADD) && isnegativeterm(cadr(p)))
		return 1;
	else if (isnegativeterm(p))
		return 1;
	else
		return 0;
}

// returns 1 if there's a symbol somewhere

int
issymbolic(U *p)
{
	if (issymbol(p))
		return 1;
	else {
		while (iscons(p)) {
			if (issymbolic(car(p)))
				return 1;
			p = cdr(p);
		}
		return 0;
	}
}

// i.e. 2, 2^3, etc.

int
isintegerfactor(U *p)
{
	if (isinteger(p) || (car(p) == symbol(POWER)
	&& isinteger(cadr(p))
	&& isinteger(caddr(p))))
		return 1;
	else
		return 0;
}

int
isoneover(U *p)
{
	if (car(p) == symbol(POWER)
	&& isminusone(caddr(p)))
		return 1;
	else
		return 0;
}

int
isfraction(U *p)
{
	if (p->k == NUM && !MEQUAL(p->u.q.b, 1))
		return 1;
	else
		return 0;
}

int
equaln(U *p, int n)
{
	switch (p->k) {
	case NUM:
		if (MEQUAL(p->u.q.a, n) && MEQUAL(p->u.q.b, 1))
			return 1;
		break;
	case DOUBLE:
		if (p->u.d == (double) n)
			return 1;
		break;
	default:
		break;
	}
	return 0;
}

int
equalq(U *p, int a, int b)
{
	switch (p->k) {
	case NUM:
		if (MEQUAL(p->u.q.a, a) && MEQUAL(p->u.q.b, b))
			return 1;
		break;
	case DOUBLE:
		if (p->u.d == (double) a / b)
			return 1;
		break;
	default:
		break;
	}
	return 0;
}

// p == 1/sqrt(2) ?

int
isoneoversqrttwo(U *p)
{
	if (car(p) == symbol(POWER)
	&& equaln(cadr(p), 2)
	&& equalq(caddr(p), -1, 2))
		return 1;
	else
		return 0;
}

// p == -1/sqrt(2) ?

int
isminusoneoversqrttwo(U *p)
{
	if (car(p) == symbol(MULTIPLY)
	&& equaln(cadr(p), -1)
	&& isoneoversqrttwo(caddr(p))
	&& length(p) == 3)
		return 1;
	else
		return 0;
}

int
isfloating(U *p)
{
	if (p->k == DOUBLE)
		return 1;
	while (iscons(p)) {
		if (isfloating(car(p)))
			return 1;
		p = cdr(p);
	}
	return 0;
}

int
isimaginaryunit(U *p)
{
	if (equal(p, imaginaryunit))
		return 1;
	else
		return 0;
}

// n/2 * i * pi ?

// return value:

//	0	no

//	1	1

//	2	-1

//	3	i

//	4	-i

int
isquarterturn(U *p)
{
	int n, minussign = 0;
	if (car(p) != symbol(MULTIPLY))
		return 0;
	if (equal(cadr(p), imaginaryunit)) {
		if (caddr(p) != symbol(PI))
			return 0;
		if (length(p) != 3)
			return 0;
		return 2;
	}
	if (!isnum(cadr(p)))
		return 0;
	if (!equal(caddr(p), imaginaryunit))
		return 0;
	if (cadddr(p) != symbol(PI))
		return 0;
	if (length(p) != 4)
		return 0;
	push(cadr(p));
	push_integer(2);
	multiply();
	n = pop_integer();
	if (n == (int) 0x80000000)
		return 0;
	if (n < 1) {
		minussign = 1;
		n = -n;
	}
	switch (n % 4) {
	case 0:
		n = 1;
		break;
	case 1:
		if (minussign)
			n = 4;
		else
			n = 3;
		break;
	case 2:
		n = 2;
		break;
	case 3:
		if (minussign)
			n = 3;
		else
			n = 4;
		break;
	}
	return n;
}

// special multiple of pi?

// returns for the following multiples of pi...

//	-4/2	-3/2	-2/2	-1/2	1/2	2/2	3/2	4/2

//	4	1	2	3	1	2	3	4

int
isnpi(U *p)
{
	int n;
	if (p == symbol(PI))
		return 2;
	if (car(p) == symbol(MULTIPLY)
	&& isnum(cadr(p))
	&& caddr(p) == symbol(PI)
	&& length(p) == 3)
		;
	else
		return 0;
	push(cadr(p));
	push_integer(2);
	multiply();
	n = pop_integer();
	if (n == (int) 0x80000000)
		return 0;
	if (n < 0)
		n = 4 - (-n) % 4;
	else
		n = 1 + (n - 1) % 4;
	return n;
}

void
eval_isprime(void)
{
	push(cadr(p1));
	eval();
	p1 = pop();
	if (isnonnegativeinteger(p1) && mprime(p1->u.q.a))
		push_integer(1);
	else
		push_integer(0);
}

/* Table of integrals

The symbol f is just a dummy symbol for creating a list f(A,B,C,C,...) where

	A	is the template expression

	B	is the result expression

	C	is an optional list of conditional expressions
*/

char *itab[] = {
// 1
	"f(a,a*x)",
// 9 (need a caveat for 7 so we can put 9 after 7)
	"f(1/x,log(x))",
// 7
	"f(x^a,x^(a+1)/(a+1))",
// 12
	"f(exp(a*x),1/a*exp(a*x))",
	"f(exp(a*x+b),1/a*exp(a*x+b))",
	"f(x*exp(a*x^2),exp(a*x^2)/(2*a))",
	"f(x*exp(a*x^2+b),exp(a*x^2+b)/(2*a))",
// 14
	"f(log(a*x),x*log(a*x)-x)",
// 15
	"f(a^x,a^x/log(a),or(not(number(a)),a>0))",
// 16
	"f(1/(a+x^2),1/sqrt(a)*arctan(x/sqrt(a)),or(not(number(a)),a>0))",
// 17
	"f(1/(a-x^2),1/sqrt(a)*arctanh(x/sqrt(a)))",
// 19
	"f(1/sqrt(a-x^2),arcsin(x/(sqrt(a))))",
// 20
	"f(1/sqrt(a+x^2),log(x+sqrt(a+x^2)))",
// 27
	"f(1/(a+b*x),1/b*log(a+b*x))",
// 28
	"f(1/(a+b*x)^2,-1/(b*(a+b*x)))",
// 29
	"f(1/(a+b*x)^3,-1/(2*b)*1/(a+b*x)^2)",
// 30
	"f(x/(a+b*x),x/b-a*log(a+b*x)/b/b)",
// 31
	"f(x/(a+b*x)^2,1/b^2*(log(a+b*x)+a/(a+b*x)))",
// 33
	"f(x^2/(a+b*x),1/b^2*(1/2*(a+b*x)^2-2*a*(a+b*x)+a^2*log(a+b*x)))",
// 34
	"f(x^2/(a+b*x)^2,1/b^3*(a+b*x-2*a*log(a+b*x)-a^2/(a+b*x)))",
// 35
	"f(x^2/(a+b*x)^3,1/b^3*(log(a+b*x)+2*a/(a+b*x)-1/2*a^2/(a+b*x)^2))",
// 37
	"f(1/x*1/(a+b*x),-1/a*log((a+b*x)/x))",
// 38
	"f(1/x*1/(a+b*x)^2,1/a*1/(a+b*x)-1/a^2*log((a+b*x)/x))",
// 39
	"f(1/x*1/(a+b*x)^3,1/a^3*(1/2*((2*a+b*x)/(a+b*x))^2+log(x/(a+b*x))))",
// 40
	"f(1/x^2*1/(a+b*x),-1/(a*x)+b/a^2*log((a+b*x)/x))",
// 41
	"f(1/x^3*1/(a+b*x),(2*b*x-a)/(2*a^2*x^2)+b^2/a^3*log(x/(a+b*x)))",
// 42
	"f(1/x^2*1/(a+b*x)^2,-(a+2*b*x)/(a^2*x*(a+b*x))+2*b/a^3*log((a+b*x)/x))",
// 60
	"f(1/(a+b*x^2),1/sqrt(a*b)*arctan(x*sqrt(a*b)/a),or(not(number(a*b)),a*b>0))",
// 61
	"f(1/(a+b*x^2),1/(2*sqrt(-a*b))*log((a+x*sqrt(-a*b))/(a-x*sqrt(-a*b))),or(not(number(a*b)),a*b<0))",
// 62 is the same as 60
// 63
	"f(x/(a+b*x^2),1/2*1/b*log(a+b*x^2))",
//64
	"f(x^2/(a+b*x^2),x/b-a/b*integral(1/(a+b*x^2),x))",
//65
	"f(1/(a+b*x^2)^2,x/(2*a*(a+b*x^2))+1/2*1/a*integral(1/(a+b*x^2),x))",
//66 is covered by 61
//70
	"f(1/x*1/(a+b*x^2),1/2*1/a*log(x^2/(a+b*x^2)))",
//71
	"f(1/x^2*1/(a+b*x^2),-1/(a*x)-b/a*integral(1/(a+b*x^2),x))",
//74
	"f(1/(a+b*x^3),1/3*1/a*(a/b)^(1/3)*(1/2*log(((a/b)^(1/3)+x)^3/(a+b*x^3))+sqrt(3)*arctan((2*x-(a/b)^(1/3))*(a/b)^(-1/3)/sqrt(3))))",
//76
	"f(x^2/(a+b*x^3),1/3*1/b*log(a+b*x^3))",
//77
	"f(1/(a+b*x^4),1/2*1/a*(a/b/4)^(1/4)*(1/2*log((x^2+2*(a/b/4)^(1/4)*x+2*(a/b/4)^(1/2))/(x^2-2*(a/b/4)^(1/4)*x+2*(a/b/4)^(1/2)))+arctan(2*(a/b/4)^(1/4)*x/(2*(a/b/4)^(1/2)-x^2))),or(not(number(a*b)),a*b>0))",
//78
	"f(1/(a+b*x^4),1/2*(-a/b)^(1/4)/a*(1/2*log((x+(-a/b)^(1/4))/(x-(-a/b)^(1/4)))+arctan(x*(-a/b)^(-1/4))),or(not(number(a*b)),a*b<0))",
//79
	"f(x/(a+b*x^4),1/2*sqrt(b/a)/b*arctan(x^2*sqrt(b/a)),or(not(number(a*b)),a*b>0))",
//80
	"f(x/(a+b*x^4),1/4*sqrt(-b/a)/b*log((x^2-sqrt(-a/b))/(x^2+sqrt(-a/b))),or(not(number(a*b)),a*b<0))",
//81
	"f(x^2/(a+b*x^4),1/4*1/b*(a/b/4)^(-1/4)*(1/2*log((x^2-2*(a/b/4)^(1/4)*x+2*sqrt(a/b/4))/(x^2+2*(a/b/4)^(1/4)*x+2*sqrt(a/b/4)))+arctan(2*(a/b/4)^(1/4)*x/(2*sqrt(a/b/4)-x^2))),or(not(number(a*b)),a*b>0))",
//82
	"f(x^2/(a+b*x^4),1/4*1/b*(-a/b)^(-1/4)*(log((x-(-a/b)^(1/4))/(x+(-a/b)^(1/4)))+2*arctan(x*(-a/b)^(-1/4))),or(not(number(a*b)),a*b<0))",
//83
	"f(x^3/(a+b*x^4),1/4*1/b*log(a+b*x^4))",
//124
	"f(sqrt(a+b*x),2/3*1/b*sqrt((a+b*x)^3))",
//125
	"f(x*sqrt(a+b*x),-2*(2*a-3*b*x)*sqrt((a+b*x)^3)/15/b^2)",
//126
	"f(x^2*sqrt(a+b*x),2*(8*a^2-12*a*b*x+15*b^2*x^2)*sqrt((a+b*x)^3)/105/b^3)",
//128
	"f(sqrt(a+b*x)/x,2*sqrt(a+b*x)+a*integral(1/x*1/sqrt(a+b*x),x))",
//129
	"f(sqrt(a+b*x)/x^2,-sqrt(a+b*x)/x+b/2*integral(1/x*1/sqrt(a+b*x),x))",
//131
	"f(1/sqrt(a+b*x),2*sqrt(a+b*x)/b)",
//132
	"f(x/sqrt(a+b*x),-2/3*(2*a-b*x)*sqrt(a+b*x)/b^2)",
//133
	"f(x^2/sqrt(a+b*x),2/15*(8*a^2-4*a*b*x+3*b^2*x^2)*sqrt(a+b*x)/b^3)",
//135
	"f(1/x*1/sqrt(a+b*x),1/sqrt(a)*log((sqrt(a+b*x)-sqrt(a))/(sqrt(a+b*x)+sqrt(a))),or(not(number(a)),a>0))",
//136
	"f(1/x*1/sqrt(a+b*x),2/sqrt(-a)*arctan(sqrt(-(a+b*x)/a)),or(not(number(a)),a<0))",
//137
	"f(1/x^2*1/sqrt(a+b*x),-sqrt(a+b*x)/a/x-1/2*b/a*integral(1/x*1/sqrt(a+b*x),x))",
//156
	"f(sqrt(x^2+a),1/2*(x*sqrt(x^2+a)+a*log(x+sqrt(x^2+a))))",
//157
	"f(1/sqrt(x^2+a),log(x+sqrt(x^2+a)))",
//158
	"f(1/x*1/sqrt(x^2+a),arcsec(x/sqrt(-a))/sqrt(-a),or(not(number(a)),a<0))",
//159
	"f(1/x*1/sqrt(x^2+a),-1/sqrt(a)*log((sqrt(a)+sqrt(x^2+a))/x),or(not(number(a)),a>0))",
//160
	"f(sqrt(x^2+a)/x,sqrt(x^2+a)-sqrt(a)*log((sqrt(a)+sqrt(x^2+a))/x),or(not(number(a)),a>0))",
//161
	"f(sqrt(x^2+a)/x,sqrt(x^2+a)-sqrt(-a)*arcsec(x/sqrt(-a)),or(not(number(a)),a<0))",
//162
	"f(x/sqrt(x^2+a),sqrt(x^2+a))",
//163
	"f(x*sqrt(x^2+a),1/3*sqrt((x^2+a)^3))",
//164 need an unexpanded version?
	"f(sqrt(a+x^6+3*a^(1/3)*x^4+3*a^(2/3)*x^2),1/4*(x*sqrt((x^2+a^(1/3))^3)+3/2*a^(1/3)*x*sqrt(x^2+a^(1/3))+3/2*a^(2/3)*log(x+sqrt(x^2+a^(1/3)))))",
	// match doesn't work for the following
	"f(sqrt(-a+x^6-3*a^(1/3)*x^4+3*a^(2/3)*x^2),1/4*(x*sqrt((x^2-a^(1/3))^3)-3/2*a^(1/3)*x*sqrt(x^2-a^(1/3))+3/2*a^(2/3)*log(x+sqrt(x^2-a^(1/3)))))",
//165
	"f(1/sqrt(a+x^6+3*a^(1/3)*x^4+3*a^(2/3)*x^2),x/a^(1/3)/sqrt(x^2+a^(1/3)))",
//166
	"f(x/sqrt(a+x^6+3*a^(1/3)*x^4+3*a^(2/3)*x^2),-1/sqrt(x^2+a^(1/3)))",
//167
	"f(x*sqrt(a+x^6+3*a^(1/3)*x^4+3*a^(2/3)*x^2),1/5*sqrt((x^2+a^(1/3))^5))",
//168
	"f(x^2*sqrt(x^2+a),1/4*x*sqrt((x^2+a)^3)-1/8*a*x*sqrt(x^2+a)-1/8*a^2*log(x+sqrt(x^2+a)))",
//169
	"f(x^3*sqrt(x^2+a),(1/5*x^2-2/15*a)*sqrt((x^2+a)^3),and(number(a),a>0))",
//170
	"f(x^3*sqrt(x^2+a),sqrt((x^2+a)^5)/5-a*sqrt((x^2+a)^3)/3,and(number(a),a<0))",
//171
	"f(x^2/sqrt(x^2+a),1/2*x*sqrt(x^2+a)-1/2*a*log(x+sqrt(x^2+a)))",
//172
	"f(x^3/sqrt(x^2+a),1/3*sqrt((x^2+a)^3)-a*sqrt(x^2+a))",
//173
	"f(1/x^2*1/sqrt(x^2+a),-sqrt(x^2+a)/a/x)",
//174
	"f(1/x^3*1/sqrt(x^2+a),-1/2*sqrt(x^2+a)/a/x^2+1/2*log((sqrt(a)+sqrt(x^2+a))/x)/a^(3/2),or(not(number(a)),a>0))",
//175
	"f(1/x^3*1/sqrt(x^2-a),1/2*sqrt(x^2-a)/a/x^2+1/2*1/(a^(3/2))*arcsec(x/(a^(1/2))),or(not(number(a)),a>0))",
//176+
	"f(x^2*sqrt(a+x^6+3*a^(1/3)*x^4+3*a^(2/3)*x^2),1/6*x*sqrt((x^2+a^(1/3))^5)-1/24*a^(1/3)*x*sqrt((x^2+a^(1/3))^3)-1/16*a^(2/3)*x*sqrt(x^2+a^(1/3))-1/16*a*log(x+sqrt(x^2+a^(1/3))),or(not(number(a)),a>0))",
//176-
	"f(x^2*sqrt(-a-3*a^(1/3)*x^4+3*a^(2/3)*x^2+x^6),1/6*x*sqrt((x^2-a^(1/3))^5)+1/24*a^(1/3)*x*sqrt((x^2-a^(1/3))^3)-1/16*a^(2/3)*x*sqrt(x^2-a^(1/3))+1/16*a*log(x+sqrt(x^2-a^(1/3))),or(not(number(a)),a>0))",
//177+
	"f(x^3*sqrt(a+x^6+3*a^(1/3)*x^4+3*a^(2/3)*x^2),1/7*sqrt((x^2+a^(1/3))^7)-1/5*a^(1/3)*sqrt((x^2+a^(1/3))^5),or(not(number(a)),a>0))",
//177-
	"f(x^3*sqrt(-a-3*a^(1/3)*x^4+3*a^(2/3)*x^2+x^6),1/7*sqrt((x^2-a^(1/3))^7)+1/5*a^(1/3)*sqrt((x^2-a^(1/3))^5),or(not(number(a)),a>0))",
//196
	"f(1/(x-a)/sqrt(x^2-a^2),-sqrt(x^2-a^2)/a/(x-a))",
//197
	"f(1/(x+a)/sqrt(x^2-a^2),sqrt(x^2-a^2)/a/(x+a))",
//200+
	"f(sqrt(a-x^2),1/2*(x*sqrt(a-x^2)+a*arcsin(x/sqrt(abs(a)))))",
//201		(seems to be handled somewhere else)
//202
	"f(1/x*1/sqrt(a-x^2),-1/sqrt(a)*log((sqrt(a)+sqrt(a-x^2))/x),or(not(number(a)),a>0))",
//203
	"f(sqrt(a-x^2)/x,sqrt(a-x^2)-sqrt(a)*log((sqrt(a)+sqrt(a-x^2))/x),or(not(number(a)),a>0))",
//204
	"f(x/sqrt(a-x^2),-sqrt(a-x^2))",
//205
	"f(x*sqrt(a-x^2),-1/3*sqrt((a-x^2)^3))",
//210
	"f(x^2*sqrt(a-x^2),-x/4*sqrt((a-x^2)^3)+1/8*a*(x*sqrt(a-x^2)+a*arcsin(x/sqrt(a))),or(not(number(a)),a>0))",
//211
	"f(x^3*sqrt(a-x^2),(-1/5*x^2-2/15*a)*sqrt((a-x^2)^3),or(not(number(a)),a>0))",
//214
	"f(x^2/sqrt(a-x^2),-x/2*sqrt(a-x^2)+a/2*arcsin(x/sqrt(a)),or(not(number(a)),a>0))",
//215
	"f(1/x^2*1/sqrt(a-x^2),-sqrt(a-x^2)/a/x,or(not(number(a)),a>0))",
//216
	"f(sqrt(a-x^2)/x^2,-sqrt(a-x^2)/x-arcsin(x/sqrt(a)),or(not(number(a)),a>0))",
//217
	"f(sqrt(a-x^2)/x^3,-1/2*sqrt(a-x^2)/x^2+1/2*log((sqrt(a)+sqrt(a-x^2))/x)/sqrt(a),or(not(number(a)),a>0))",
//218
	"f(sqrt(a-x^2)/x^4,-1/3*sqrt((a-x^2)^3)/a/x^3,or(not(number(a)),a>0))",
// 273
	"f(sqrt(a*x^2+b),x*sqrt(a*x^2+b)/2+b*log(x*sqrt(a)+sqrt(a*x^2+b))/2/sqrt(a),and(number(a),a>0))",
// 274
	"f(sqrt(a*x^2+b),x*sqrt(a*x^2+b)/2+b*arcsin(x*sqrt(-a/b))/2/sqrt(-a),and(number(a),a<0))",
// 290
	"f(sin(a*x),-cos(a*x)/a)",
// 291
	"f(cos(a*x),sin(a*x)/a)",
// 292
	"f(tan(a*x),-log(cos(a*x))/a)",
// 293
	"f(1/tan(a*x),log(sin(a*x))/a)",
// 294
	"f(1/cos(a*x),log(tan(pi/4+a*x/2))/a)",
// 295
	"f(1/sin(a*x),log(tan(a*x/2))/a)",
// 296
	"f(sin(a*x)^2,x/2-sin(2*a*x)/(4*a))",
// 297
	"f(sin(a*x)^3,-cos(a*x)*(sin(a*x)^2+2)/(3*a))",
// 298
	"f(sin(a*x)^4,3/8*x-sin(2*a*x)/(4*a)+sin(4*a*x)/(32*a))",
// 302
	"f(cos(a*x)^2,x/2+sin(2*a*x)/(4*a))",
// 303
	"f(cos(a*x)^3,sin(a*x)*(cos(a*x)^2+2)/(3*a))",
// 304
	"f(cos(a*x)^4,3/8*x+sin(2*a*x)/(4*a)+sin(4*a*x)/(32*a))",
// 308
	"f(1/sin(a*x)^2,-1/(a*tan(a*x)))",
// 312
	"f(1/cos(a*x)^2,tan(a*x)/a)",
// 318
	"f(sin(a*x)*cos(a*x),sin(a*x)^2/(2*a))",
// 320
	"f(sin(a*x)^2*cos(a*x)^2,-sin(4*a*x)/(32*a)+x/8)",
// 326
	"f(sin(a*x)/cos(a*x)^2,1/(a*cos(a*x)))",
// 327
	"f(sin(a*x)^2/cos(a*x),(log(tan(pi/4+a*x/2))-sin(a*x))/a)",
// 328
	"f(cos(a*x)/sin(a*x)^2,-1/(a*sin(a*x)))",
// 329
	"f(1/(sin(a*x)*cos(a*x)),log(tan(a*x))/a)",
// 330
	"f(1/(sin(a*x)*cos(a*x)^2),(1/cos(a*x)+log(tan(a*x/2)))/a)",
// 331
	"f(1/(sin(a*x)^2*cos(a*x)),(log(tan(pi/4+a*x/2))-1/sin(a*x))/a)",
// 333
	"f(1/(sin(a*x)^2*cos(a*x)^2),-2/(a*tan(2*a*x)))",
// 335
	"f(sin(a+b*x),-cos(a+b*x)/b)",
// 336
	"f(cos(a+b*x),sin(a+b*x)/b)",
// 337+ (with the addition of b)
	"f(1/(b+b*sin(a*x)),-tan(pi/4-a*x/2)/a/b)",
// 337- (with the addition of b)
	"f(1/(b-b*sin(a*x)),tan(pi/4+a*x/2)/a/b)",
// 338 (with the addition of b)
	"f(1/(b+b*cos(a*x)),tan(a*x/2)/a/b)",
// 339 (with the addition of b)
	"f(1/(b-b*cos(a*x)),-1/tan(a*x/2)/a/b)",
// 340
	"f(1/(a+b*sin(x)),1/sqrt(b^2-a^2)*log((a*tan(x/2)+b-sqrt(b^2-a^2))/(a*tan(x/2)+b+sqrt(b^2-a^2))),b^2-a^2)", // check that b^2-a^2 is not zero
// 341
	"f(1/(a+b*cos(x)),1/sqrt(b^2-a^2)*log((sqrt(b^2-a^2)*tan(x/2)+a+b)/(sqrt(b^2-a^2)*tan(x/2)-a-b)),b^2-a^2)", // check that b^2-a^2 is not zero
// 389
	"f(x*sin(a*x),sin(a*x)/a^2-x*cos(a*x)/a)",
// 390
	"f(x^2*sin(a*x),2*x*sin(a*x)/a^2-(a^2*x^2-2)*cos(a*x)/a^3)",
// 393
	"f(x*cos(a*x),cos(a*x)/a^2+x*sin(a*x)/a)",
// 394
	"f(x^2*cos(a*x),2*x*cos(a*x)/a^2+(a^2*x^2-2)*sin(a*x)/a^3)",
// 441
	"f(arcsin(a*x),x*arcsin(a*x)+sqrt(1-a^2*x^2)/a)",
// 442
	"f(arccos(a*x),x*arccos(a*x)+sqrt(1-a^2*x^2)/a)",
// 443
	"f(arctan(a*x),x*arctan(a*x)-1/2*log(1+a^2*x^2)/a)",
// 485 (with addition of a)
	"f(log(a*x),x*log(a*x)-x)",
// 486 (with addition of a)
	"f(x*log(a*x),x^2*log(a*x)/2-x^2/4)",
// 487 (with addition of a)
	"f(x^2*log(a*x),x^3*log(a*x)/3-1/9*x^3)",
// 489
	"f(log(x)^2,x*log(x)^2-2*x*log(x)+2*x)",
// 493 (with addition of a)
	"f(1/x*1/(a+log(x)),log(a+log(x)))",
// 499
	"f(log(a*x+b),(a*x+b)*log(a*x+b)/a-x)",
// 500
	"f(log(a*x+b)/x^2,a/b*log(x)-(a*x+b)*log(a*x+b)/b/x)",
// 554
	"f(sinh(x),cosh(x))",
// 555
	"f(cosh(x),sinh(x))",
// 556
	"f(tanh(x),log(cosh(x)))",
// 560
	"f(x*sinh(x),x*cosh(x)-sinh(x))",
// 562
	"f(x*cosh(x),x*sinh(x)-cosh(x))",
// 566
	"f(sinh(x)^2,sinh(2*x)/4-x/2)",
// 569
	"f(tanh(x)^2,x-tanh(x))",
// 572
	"f(cosh(x)^2,sinh(2*x)/4+x/2)",
// ?
	"f(x^3*exp(a*x^2),exp(a*x^2)*(x^2/a-1/(a^2))/2)",
// ?
	"f(x^3*exp(a*x^2+b),exp(a*x^2)*exp(b)*(x^2/a-1/(a^2))/2)",
// ?
	"f(exp(a*x^2),-i*sqrt(pi)*erf(i*sqrt(a)*x)/sqrt(a)/2)",
// ?
	"f(erf(a*x),x*erf(a*x)+exp(-a^2*x^2)/a/sqrt(pi))",

// these are needed for the surface integral in the manual

	"f(x^2*(1-x^2)^(3/2),(x*sqrt(1-x^2)*(-8*x^4+14*x^2-3)+3*arcsin(x))/48)",
	"f(x^2*(1-x^2)^(5/2),(x*sqrt(1-x^2)*(48*x^6-136*x^4+118*x^2-15)+15*arcsin(x))/384)",
	"f(x^4*(1-x^2)^(3/2),(-x*sqrt(1-x^2)*(16*x^6-24*x^4+2*x^2+3)+3*arcsin(x))/128)",

	"f(x*exp(a*x),exp(a*x)*(a*x-1)/(a^2))",
	"f(x*exp(a*x+b),exp(a*x+b)*(a*x-1)/(a^2))",

	"f(x^2*exp(a*x),exp(a*x)*(a^2*x^2-2*a*x+2)/(a^3))",
	"f(x^2*exp(a*x+b),exp(a*x+b)*(a^2*x^2-2*a*x+2)/(a^3))",

	"f(x^3*exp(a*x),exp(a*x)*x^3/a-3/a*integral(x^2*exp(a*x),x))",
	"f(x^3*exp(a*x+b),exp(a*x+b)*x^3/a-3/a*integral(x^2*exp(a*x+b),x))",

	NULL,
};

/* Laguerre function

The computation uses the following recurrence relation.

	L(x,0,k) = 1

	L(x,1,k) = -x + k + 1

	n*L(x,n,k) = (2*(n-1)+1-x+k)*L(x,n-1,k) - (n-1+k)*L(x,n-2,k)

In the "for" loop i = n-1 so the recurrence relation becomes

	(i+1)*L(x,n,k) = (2*i+1-x+k)*L(x,n-1,k) - (i+k)*L(x,n-2,k)
*/

void
eval_laguerre(void)
{
	// 1st arg
	push(cadr(p1));
	eval();
	// 2nd arg
	push(caddr(p1));
	eval();
	// 3rd arg
	push(cadddr(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		push_integer(0);
	else
		push(p2);
	laguerre();
}

#undef X
#undef N
#undef K
#undef Y
#undef Y0
#undef Y1

#define X p1
#define N p2
#define K p3
#define Y p4
#define Y0 p5
#define Y1 p6

void
laguerre(void)
{
	int n;
	save();
	K = pop();
	N = pop();
	X = pop();
	push(N);
	n = pop_integer();
	if (n < 0) {
		push_symbol(LAGUERRE);
		push(X);
		push(N);
		push(K);
		list(4);
		restore();
		return;
	}
	if (issymbol(X))
		laguerre2(n);
	else {
		Y = X;			// do this when X is an expr
		X = symbol(SPECX);
		laguerre2(n);
		X = Y;
		push_symbol(SPECX);
		push(X);
		subst();
		eval();
	}
	restore();
}

void
laguerre2(int n)
{
	int i;
	push_integer(1);
	push_integer(0);
	Y1 = pop();
	for (i = 0; i < n; i++) {
		Y0 = Y1;
		Y1 = pop();
		push_integer(2 * i + 1);
		push(X);
		subtract();
		push(K);
		add();
		push(Y1);
		multiply();
		push_integer(i);
		push(K);
		add();
		push(Y0);
		multiply();
		subtract();
		push_integer(i + 1);
		divide();
	}
}

// Find the least common multiple of two expressions.

void
eval_lcm(void)
{
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		lcm();
		p1 = cdr(p1);
	}
}

void
lcm(void)
{
	int x;
	x = expanding;
	save();
	yylcm();
	restore();
	expanding = x;
}

void
yylcm(void)
{
	expanding = 1;
	p2 = pop();
	p1 = pop();
	push(p1);
	push(p2);
	gcd();
	push(p1);
	divide();
	push(p2);
	divide();
	inverse();
}

/* Return the leading coefficient of a polynomial.

Example

	leading(5x^2+x+1,x)

Result

	5

The result is undefined if P is not a polynomial. */

void
eval_leading(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	p1 = pop();
	if (p1 == symbol(NIL))
		guess();
	else
		push(p1);
	leading();
}

#undef P
#undef X
#undef N

#define P p1
#define X p2
#define N p3

void
leading(void)
{
	save();
	X = pop();
	P = pop();
	push(P);	// N = degree of P
	push(X);
	degree();
	N = pop();
	push(P);	// divide through by X ^ N
	push(X);
	push(N);
	power();
	divide();
	push(X);	// remove terms that depend on X
	filter();
	restore();
}

/* Legendre function

The computation uses the following recurrence relation.

	P(x,0) = 1

	P(x,1) = x

	n*P(x,n) = (2*(n-1)+1)*x*P(x,n-1) - (n-1)*P(x,n-2)

In the "for" loop we have i = n-1 so the recurrence relation becomes

	(i+1)*P(x,n) = (2*i+1)*x*P(x,n-1) - i*P(x,n-2)

For m > 0

	P(x,n,m) = (-1)^m * (1-x^2)^(m/2) * d^m/dx^m P(x,n)
*/

void
eval_legendre(void)
{
	// 1st arg
	push(cadr(p1));
	eval();
	// 2nd arg
	push(caddr(p1));
	eval();
	// 3rd arg (optional)
	push(cadddr(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		push_integer(0);
	else
		push(p2);
	legendre();
}

#undef X
#undef N
#undef M
#undef Y
#undef Y0
#undef Y1

#define X p1
#define N p2
#define M p3
#define Y p4
#define Y0 p5
#define Y1 p6

void
legendre(void)
{
	save();
	legendre_nib();
	restore();
}

void
legendre_nib(void)
{
	int m, n;
	M = pop();
	N = pop();
	X = pop();
	push(N);
	n = pop_integer();
	push(M);
	m = pop_integer();
	if (n < 0 || m < 0) {
		push_symbol(LEGENDRE);
		push(X);
		push(N);
		push(M);
		list(4);
		return;
	}
	if (issymbol(X))
		legendre2(n, m);
	else {
		Y = X;			// do this when X is an expr
		X = symbol(SPECX);
		legendre2(n, m);
		X = Y;
		push_symbol(SPECX);
		push(X);
		subst();
		eval();
	}
	legendre3(m);
}

void
legendre2(int n, int m)
{
	int i;
	push_integer(1);
	push_integer(0);
	Y1 = pop();
	//	i=1	Y0 = 0
	//		Y1 = 1
	//		((2*i+1)*x*Y1 - i*Y0) / i = x
	//
	//	i=2	Y0 = 1
	//		Y1 = x
	//		((2*i+1)*x*Y1 - i*Y0) / i = -1/2 + 3/2*x^2
	//
	//	i=3	Y0 = x
	//		Y1 = -1/2 + 3/2*x^2
	//		((2*i+1)*x*Y1 - i*Y0) / i = -3/2*x + 5/2*x^3
	for (i = 0; i < n; i++) {
		Y0 = Y1;
		Y1 = pop();
		push_integer(2 * i + 1);
		push(X);
		multiply();
		push(Y1);
		multiply();
		push_integer(i);
		push(Y0);
		multiply();
		subtract();
		push_integer(i + 1);
		divide();
	}
	for (i = 0; i < m; i++) {
		push(X);
		derivative();
	}
}

// tos = tos * (-1)^m * (1-x^2)^(m/2)

void
legendre3(int m)
{
	if (m == 0)
		return;
	if (car(X) == symbol(COS)) {
		push(cadr(X));
		sine();
		square();
	} else if (car(X) == symbol(SIN)) {
		push(cadr(X));
		cosine();
		square();
	} else {
		push_integer(1);
		push(X);
		square();
		subtract();
	}
	push_integer(m);
	push_rational(1, 2);
	multiply();
	power();
	multiply();
	if (m % 2)
		negate();
}

// Create a list from n things on the stack.

void
list(int n)
{
	int i;
	push(symbol(NIL));
	for (i = 0; i < n; i++)
		cons();
}

// natural logarithm

void
eval_log(void)
{
	push(cadr(p1));
	eval();
	logarithm();
}

void
logarithm(void)
{
	save();
	yylog();
	restore();
}

void
yylog(void)
{
	double d;
	p1 = pop();
	if (p1 == symbol(EXP1)) {
		push_integer(1);
		return;
	}
	if (equaln(p1, 1)) {
		push_integer(0);
		return;
	}
	if (isnegativenumber(p1)) {
		push(p1);
		negate();
		logarithm();
		push(imaginaryunit);
		push_symbol(PI);
		multiply();
		add();
		return;
	}
	if (isdouble(p1)) {
		d = log(p1->u.d);
		push_double(d);
		return;
	}
	// rational number and not an integer?
	if (isfraction(p1)) {
		push(p1);
		numerator();
		logarithm();
		push(p1);
		denominator();
		logarithm();
		subtract();
		return;
	}
	// log(a ^ b) --> b log(a)
	if (car(p1) == symbol(POWER)) {
		push(caddr(p1));
		push(cadr(p1));
		logarithm();
		multiply();
		return;
	}
	// log(a * b) --> log(a) + log(b)
	if (car(p1) == symbol(MULTIPLY)) {
		push_integer(0);
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			logarithm();
			add();
			p1 = cdr(p1);
		}
		return;
	}
	push_symbol(LOG);
	push(p1);
	list(2);
}

// Bignum addition and subtraction

unsigned int *
madd(unsigned int *a, unsigned int *b)
{
	if (MSIGN(a) == MSIGN(b))
		return madd_nib(a, b);	// same sign, add together
	else
		return msub_nib(a, b);	// opposite sign, find difference
}

unsigned int *
msub(unsigned int *a, unsigned int *b)
{
	if (MSIGN(a) == MSIGN(b))
		return msub_nib(a, b);	// same sign, find difference
	else
		return madd_nib(a, b);	// opposite sign, add together
}

unsigned int *
madd_nib(unsigned int *a, unsigned int *b)
{
	int i, sign;
	unsigned int c, *x;
	sign = MSIGN(a);
	if (MLENGTH(a) < MLENGTH(b)) {
		x = a;
		a = b;
		b = x;
	}
	x = mnew(MLENGTH(a) + 1);
	c = 0;
	for (i = 0; i < MLENGTH(b); i++) {
		x[i] = a[i] + b[i] + c;
		if (c)
			if (a[i] >= x[i])
				c = 1;
			else
				c = 0;
		else
			if (a[i] > x[i])
				c = 1;
			else
				c = 0;
	}
	for (i = MLENGTH(b); i < MLENGTH(a); i++) {
		x[i] = a[i] + c;
		if (a[i] > x[i])
			c = 1;
		else
			c = 0;
	}
	x[MLENGTH(a)] = c;
	for (i = MLENGTH(a); i > 0; i--)
		if (x[i])
			break;
	MLENGTH(x) = i + 1;
	MSIGN(x) = sign;
	return x;
}

unsigned int *
msub_nib(unsigned int *a, unsigned int *b)
{
	int i, sign = 0;
	unsigned int c, *x;
	switch (add_ucmp(a, b)) {
	case 0:
		return mint(0);
	case 1:
		sign = MSIGN(a);	/* |a| > |b| */
		break;
	case -1:
		sign = -MSIGN(a);	/* |a| < |b| */
		x = a;
		a = b;
		b = x;
		break;
	}
	x = mnew(MLENGTH(a));
	c = 0;
	for (i = 0; i < MLENGTH(b); i++) {
		x[i] = a[i] - b[i] - c;
		if (c)
			if (a[i] <= x[i])
				c = 1;
			else
				c = 0;
		else
			if (a[i] < x[i])
				c = 1;
			else
				c = 0;
	}
	for (i = MLENGTH(b); i < MLENGTH(a); i++) {
		x[i] = a[i] - c;
		if (a[i] < x[i])
			c = 1;
		else
			c = 0;
	}
	for (i = MLENGTH(a) - 1; i > 0; i--)
		if (x[i])
			break;
	MLENGTH(x) = i + 1;
	MSIGN(x) = sign;
	return x;
}

// unsigned compare

int
add_ucmp(unsigned int *a, unsigned int *b)
{
	int i;
	if (MLENGTH(a) < MLENGTH(b))
		return -1;
	if (MLENGTH(a) > MLENGTH(b))
		return 1;
	for (i = MLENGTH(a) - 1; i > 0; i--)
		if (a[i] != b[i])
			break;
	if (a[i] < b[i])
		return -1;
	if (a[i] > b[i])
		return 1;
	return 0;
}

/* Magnitude of complex z

	z		mag(z)
	-		------

	a		a

	-a		a

	(-1)^a		1

	exp(a + i b)	exp(a)

	a b		mag(a) mag(b)

	a + i b		sqrt(a^2 + b^2)
*/

void
eval_mag(void)
{
	push(cadr(p1));
	eval();
	mag();
}

void
mag(void)
{
	save();
	p1 = pop();
	push(p1);
	numerator();
	yymag();
	push(p1);
	denominator();
	yymag();
	divide();
	restore();
}

void
yymag(void)
{
	save();
	p1 = pop();
	if (isnegativenumber(p1)) {
		push(p1);
		negate();
	} else if (car(p1) == symbol(POWER) && equaln(cadr(p1), -1))
		// -1 to a power
		push_integer(1);
	else if (car(p1) == symbol(POWER) && cadr(p1) == symbol(EXP1)) {
		// exponential
		push(caddr(p1));
		real();
		exponential();
	} else if (car(p1) == symbol(MULTIPLY)) {
		// product
		push_integer(1);
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			mag();
			multiply();
			p1 = cdr(p1);
		}
	} else if (car(p1) == symbol(ADD)) {
		// sum
		push(p1);
		rect(); // convert polar terms, if any
		p1 = pop();
		push(p1);
		real();
		push_integer(2);
		power();
		push(p1);
		imag();
		push_integer(2);
		power();
		add();
		push_rational(1, 2);
		power();
		simplify_trig();
	} else
		// default (all real)
		push(p1);
	restore();
}

/*	The starting point for a symbolic computation is in run.c

	Input is scanned in scan.c

	Expression evaluation is done in eval.c

	Output is formatted in display.c
*/

int
main(int argc, char *argv[])
{
	static char buf[1000];
	init();
	clear();
	if (argc > 1)
		run_script(argv[1]);
	for (;;) {
		printf("? ");
		fgets(buf, sizeof buf, stdin);
		run(buf);
	}
	return 0;
}

void
run_script(char *filename)
{
	int fd, n;
	char *buf;
	fd = open(filename, O_RDONLY, 0);
	if (fd == -1) {
		printf("cannot open %s\n", filename);
		exit(1);
	}
	// get file size
	n = lseek(fd, 0, SEEK_END);
	if (n == -1) {
		printf("lseek kaput\n");
		exit(1);
	}
	lseek(fd, 0, SEEK_SET);
	buf = malloc(n + 1);
	if (buf == NULL) {
		printf("malloc %d kaput\n", n + 1);
		exit(1);
	}
	if (read(fd, buf, n) != n) {
		printf("read kaput\n");
		exit(1);
	}
	close(fd);
	buf[n] = 0;
	run(buf);
	free(buf);
}

void
printstr(char *s)
{
	while (*s)
		printchar(*s++);
}

void
printchar(int c)
{
	fputc(c, stdout);
}

void
printchar_nowrap(int c)
{
	printchar(c);
}

void
eval_draw(void)
{
	push(symbol(NIL));
}

void
eval_sample(void)
{
}

void
clear_display(void)
{
}

void
cmdisplay(void)
{
	display();
}

// Bignum compare
//
//	returns
//
//	-1		a < b
//
//	0		a = b
//
//	1		a > b

int
mcmp(unsigned int *a, unsigned int *b)
{
	int i;
	if (MSIGN(a) == -1 && MSIGN(b) == 1)
		return -1;
	if (MSIGN(a) == 1 && MSIGN(b) == -1)
		return 1;
	// same sign
	if (MLENGTH(a) < MLENGTH(b)) {
		if (MSIGN(a) == 1)
			return -1;
		else
			return 1;
	}
	if (MLENGTH(a) > MLENGTH(b)) {
		if (MSIGN(a) == 1)
			return 1;
		else
			return -1;
	}
	// same length
	for (i = MLENGTH(a) - 1; i > 0; i--)
		if (a[i] != b[i])
			break;
	if (a[i] < b[i]) {
		if (MSIGN(a) == 1)
			return -1;
		else
			return 1;
	}
	if (a[i] > b[i]) {
		if (MSIGN(a) == 1)
			return 1;
		else
			return -1;
	}
	return 0;
}

int
mcmpint(unsigned int *a, int n)
{
	int t;
	unsigned int *b;
	b = mint(n);
	t = mcmp(a, b);
	mfree(b);
	return t;
}

//-----------------------------------------------------------------------------
//
//	Bignum GCD
//
//	Uses the binary GCD algorithm.
//
//	See "The Art of Computer Programming" p. 338.
//
//	mgcd always returns a positive value
//
//	mgcd(0, 0) = 0
//
//	mgcd(u, 0) = |u|
//
//	mgcd(0, v) = |v|
//
//-----------------------------------------------------------------------------

unsigned int *
mgcd(unsigned int *u, unsigned int *v)
{
	int i, k, n;
	unsigned int *t;
	if (MZERO(u)) {
		t = mcopy(v);
		MSIGN(t) = 1;
		return t;
	}
	if (MZERO(v)) {
		t = mcopy(u);
		MSIGN(t) = 1;
		return t;
	}
	u = mcopy(u);
	v = mcopy(v);
	MSIGN(u) = 1;
	MSIGN(v) = 1;
	k = 0;
	while ((u[0] & 1) == 0 && (v[0] & 1) == 0) {
		mshiftright(u);
		mshiftright(v);
		k++;
	}
	if (u[0] & 1) {
		t = mcopy(v);
		MSIGN(t) *= -1;
	} else
		t = mcopy(u);
	while (1) {
		while ((t[0] & 1) == 0)
			mshiftright(t);
		if (MSIGN(t) == 1) {
			mfree(u);
			u = mcopy(t);
		} else {
			mfree(v);
			v = mcopy(t);
			MSIGN(v) *= -1;
		}
		mfree(t);
		t = msub(u, v);
		if (MZERO(t)) {
			mfree(t);
			mfree(v);
			n = (k / 32) + 1;
			v = mnew(n);
			MSIGN(v) = 1;
			MLENGTH(v) = n;
			for (i = 0; i < n; i++)
				v[i] = 0;
			mp_set_bit(v, k);
			t = mmul(u, v);
			mfree(u);
			mfree(v);
			return t;
		}
	}
}

void
new_string(char *s)
{
	save();
	p1 = alloc();
	p1->k = STR;
	p1->u.str = strdup(s);
	push(p1);
	restore();
}

void
out_of_memory(void)
{
	stop("out of memory");
}

void
push_zero_matrix(int i, int j)
{
	push(alloc_tensor(i * j));
	stack[tos - 1]->u.tensor->ndim = 2;
	stack[tos - 1]->u.tensor->dim[0] = i;
	stack[tos - 1]->u.tensor->dim[1] = j;
}

void
push_identity_matrix(int n)
{
	int i;
	push_zero_matrix(n, n);
	for (i = 0; i < n; i++)
		stack[tos - 1]->u.tensor->elem[i * n + i] = one;
}

void
push_cars(U *p)
{
	while (iscons(p)) {
		push(car(p));
		p = cdr(p);
	}
}

void
peek(void)
{
	save();
	p1 = pop();
	push(p1);
	print(p1);
	restore();
}

void
peek2(void)
{
	print_lisp(stack[tos - 2]);
	print_lisp(stack[tos - 1]);
}

int
equal(U *p1, U *p2)
{
	if (cmp_expr(p1, p2) == 0)
		return 1;
	else
		return 0;
}

int
lessp(U *p1, U *p2)
{
	if (cmp_expr(p1, p2) < 0)
		return 1;
	else
		return 0;
}

int
sign(int n)
{
	if (n < 0)
		return -1;
	else if (n > 0)
		return 1;
	else
		return 0;
}

int
cmp_expr(U *p1, U *p2)
{
	int n;
	if (p1 == p2)
		return 0;
	if (p1 == symbol(NIL))
		return -1;
	if (p2 == symbol(NIL))
		return 1;
	if (isnum(p1) && isnum(p2))
		return sign(compare_numbers(p1, p2));
	if (isnum(p1))
		return -1;
	if (isnum(p2))
		return 1;
	if (isstr(p1) && isstr(p2))
		return sign(strcmp(p1->u.str, p2->u.str));
	if (isstr(p1))
		return -1;
	if (isstr(p2))
		return 1;
	if (issymbol(p1) && issymbol(p2))
		return sign(strcmp(get_printname(p1), get_printname(p2)));
	if (issymbol(p1))
		return -1;
	if (issymbol(p2))
		return 1;
	if (istensor(p1) && istensor(p2))
		return compare_tensors(p1, p2);
	if (istensor(p1))
		return -1;
	if (istensor(p2))
		return 1;
	while (iscons(p1) && iscons(p2)) {
		n = cmp_expr(car(p1), car(p2));
		if (n != 0)
			return n;
		p1 = cdr(p1);
		p2 = cdr(p2);
	}
	if (iscons(p2))
		return -1;
	if (iscons(p1))
		return 1;
	return 0;
}

int
length(U *p)
{
	int n = 0;
	while (iscons(p)) {
		p = cdr(p);
		n++;
	}
	return n;
}

U *
unique(U *p)
{
	save();
	p1 = symbol(NIL);
	p2 = symbol(NIL);
	unique_f(p);
	if (p2 != symbol(NIL))
		p1 = symbol(NIL);
	p = p1;
	restore();
	return p;
}

void
unique_f(U *p)
{
	if (isstr(p)) {
		if (p1 == symbol(NIL))
			p1 = p;
		else if (p != p1)
			p2 = p;
		return;
	}
	while (iscons(p)) {
		unique_f(car(p));
		if (p2 != symbol(NIL))
			return;
		p = cdr(p);
	}
}

void
ssqrt(void)
{
	push_rational(1, 2);
	power();
}

void
yyexpand(void)
{
	int x;
	x = expanding;
	expanding = 1;
	eval();
	expanding = x;
}

void
exponential(void)
{
	push_symbol(EXP1);
	swap();
	power();
}

void
square(void)
{
	push_integer(2);
	power();
}

int
sort_stack_cmp(const void *p1, const void *p2)
{
	return cmp_expr(*((U **) p1), *((U **) p2));
}

void
sort_stack(int n)
{
	qsort(stack + tos - n, n, sizeof (U *), sort_stack_cmp);
}

// Bignum modular power (x^n mod m)

// could do indexed bit test instead of shift right

unsigned int *
mmodpow(unsigned int *x, unsigned int *n, unsigned int *m)
{
	unsigned int *y, *z;
	x = mcopy(x);
	n = mcopy(n);
	y = mint(1);
	while (1) {
		if (n[0] & 1) {
			z = mmul(y, x);
			mfree(y);
			y = mmod(z, m);
			mfree(z);
		}
		mshiftright(n);
		if (MZERO(n))
			break;
		z = mmul(x, x);
		mfree(x);
		x = mmod(z, m);
		mfree(z);
	}
	mfree(x);
	mfree(n);
	return y;
}

// Bignum multiplication and division

unsigned int *
mmul(unsigned int *a, unsigned int *b)
{
	int alen, blen, i, n;
	unsigned int *t, *x;
	if (MZERO(a) || MZERO(b))
		return mint(0);
	if (MLENGTH(a) == 1 && a[0] == 1) {
		t = mcopy(b);
		MSIGN(t) *= MSIGN(a);
		return t;
	}
	if (MLENGTH(b) == 1 && b[0] == 1) {
		t = mcopy(a);
		MSIGN(t) *= MSIGN(b);
		return t;
	}
	alen = MLENGTH(a);
	blen = MLENGTH(b);
	n = alen + blen;
	x = mnew(n);
	t = mnew(alen + 1);
	for (i = 0; i < n; i++)
		x[i] = 0;
	/* sum of partial products */
	for (i = 0; i < blen; i++) {
		mulf(t, a, alen, b[i]);
		addf(x + i, t, alen + 1);
	}
	mfree(t);
	/* length of product */
	for (i = n - 1; i > 0; i--)
		if (x[i])
			break;
	MLENGTH(x) = i + 1;
	MSIGN(x) = MSIGN(a) * MSIGN(b);
	return x;
}

unsigned int *
mdiv(unsigned int *a, unsigned int *b)
{
	int alen, blen, i, n;
	unsigned int c, *t, *x, *y;
	uint64_t jj, kk;
	if (MZERO(b))
		stop("divide by zero");
	if (MZERO(a))
		return mint(0);
	alen = MLENGTH(a);
	blen = MLENGTH(b);
	n = alen - blen;
	if (n < 0)
		return mint(0);
	x = mnew(alen + 1);
	for (i = 0; i < alen; i++)
		x[i] = a[i];
	x[i] = 0;
	y = mnew(n + 1);
	t = mnew(blen + 1);
	/* Add 1 here to round up in case the remaining words are non-zero. */
	kk = (uint64_t) b[blen - 1] + 1;
	for (i = 0; i <= n; i++) {
		y[n - i] = 0;
		for (;;) {
			/* estimate the partial quotient */
			jj = (uint64_t) x[alen - i - 0] << 32 | x[alen - i - 1];
			c = (uint32_t) (jj / kk); // compiler warns w/o cast
			if (c == 0) {
				if (ge(x + n - i, b, blen)) { /* see note 1 */
					y[n - i]++;
					subf(x + n - i, b, blen);
				}
				break;
			}
			y[n - i] += c;
			mulf(t, b, blen, c);
			subf(x + n - i, t, blen + 1);
		}
	}
	mfree(t);
	mfree(x);
	/* length of quotient */
	for (i = n; i > 0; i--)
		if (y[i])
			break;
	if (i == 0 && y[0] == 0) {
		mfree(y);
		y = mint(0);
	} else {
		MLENGTH(y) = i + 1;
		MSIGN(y) = MSIGN(a) * MSIGN(b);
	}
	return y;
}

// a = a + b

void
addf(unsigned int *a, unsigned int *b, int len)
{
	int i;
	long long t = 0; /* can be signed or unsigned */
	for (i = 0; i < len; i++) {
		t += (long long) a[i] + b[i];
		a[i] = (unsigned int) t;
		t >>= 32;
	}
}

// a = a - b

void
subf(unsigned int *a, unsigned int *b, int len)
{
	int i;
	long long t = 0; /* must be signed */
	for (i = 0; i < len; i++) {
		t += (long long) a[i] - b[i];
		a[i] = (unsigned int) t;
		t >>= 32;
	}
}

// a = b * c

// 0xffffffff + 0xffffffff * 0xffffffff == 0xffffffff00000000

void
mulf(unsigned int *a, unsigned int *b, int len, unsigned int c)
{
	int i;
	uint64_t t = 0; /* must be unsigned */
	for (i = 0; i < len; i++) {
		t += (uint64_t) b[i] * c;
		a[i] = (unsigned int) t;
		t >>= 32;
	}
	a[i] = (unsigned int) t;
}

unsigned int *
mmod(unsigned int *a, unsigned int *b)
{
	int alen, blen, i, n;
	unsigned int c, *t, *x, *y;
	uint64_t jj, kk;
	if (MZERO(b))
		stop("divide by zero");
	if (MZERO(a))
		return mint(0);
	alen = MLENGTH(a);
	blen = MLENGTH(b);
	n = alen - blen;
	if (n < 0)
		return mcopy(a);
	x = mnew(alen + 1);
	for (i = 0; i < alen; i++)
		x[i] = a[i];
	x[i] = 0;
	y = mnew(n + 1);
	t = mnew(blen + 1);
	kk = (uint64_t) b[blen - 1] + 1;
	for (i = 0; i <= n; i++) {
		y[n - i] = 0;
		for (;;) {
			/* estimate the partial quotient */
			jj = (uint64_t) x[alen - i - 0] << 32 | x[alen - i - 1];
			c = (uint32_t) (jj / kk); // compiler warns w/o cast
			if (c == 0) {
				if (ge(x + n - i, b, blen)) { /* see note 1 */
					y[n - i]++;
					subf(x + n - i, b, blen);
				}
				break;
			}
			y[n - i] += c;
			mulf(t, b, blen, c);
			subf(x + n - i, t, blen + 1);
		}
	}
	mfree(t);
	mfree(y);
	/* length of remainder */
	for (i = blen - 1; i > 0; i--)
		if (x[i])
			break;
	if (i == 0 && x[0] == 0) {
		mfree(x);
		x = mint(0);
	} else {
		MLENGTH(x) = i + 1;
		MSIGN(x) = MSIGN(a);
	}
	return x;
}

// return both quotient and remainder of a/b

void
mdivrem(unsigned int **q, unsigned int **r, unsigned int *a, unsigned int *b)
{
	int alen, blen, i, n;
	unsigned int c, *t, *x, *y;
	uint64_t jj, kk;
	if (MZERO(b))
		stop("divide by zero");
	if (MZERO(a)) {
		*q = mint(0);
		*r = mint(0);
		return;
	}
	alen = MLENGTH(a);
	blen = MLENGTH(b);
	n = alen - blen;
	if (n < 0) {
		*q = mint(0);
		*r = mcopy(a);
		return;
	}
	x = mnew(alen + 1);
	for (i = 0; i < alen; i++)
		x[i] = a[i];
	x[i] = 0;
	y = mnew(n + 1);
	t = mnew(blen + 1);
	kk = (uint64_t) b[blen - 1] + 1;
	for (i = 0; i <= n; i++) {
		y[n - i] = 0;
		for (;;) {
			/* estimate the partial quotient */
			jj = (uint64_t) x[alen - i - 0] << 32 | x[alen - i - 1];
			c = (uint32_t) (jj / kk); // compiler warns w/o cast
			if (c == 0) {
				if (ge(x + n - i, b, blen)) { /* see note 1 */
					y[n - i]++;
					subf(x + n - i, b, blen);
				}
				break;
			}
			y[n - i] += c;
			mulf(t, b, blen, c);
			subf(x + n - i, t, blen + 1);
		}
	}
	mfree(t);
	/* length of quotient */
	for (i = n; i > 0; i--)
		if (y[i])
			break;
	if (i == 0 && y[0] == 0) {
		mfree(y);
		y = mint(0);
	} else {
		MLENGTH(y) = i + 1;
		MSIGN(y) = MSIGN(a) * MSIGN(b);
	}
	/* length of remainder */
	for (i = blen - 1; i > 0; i--)
		if (x[i])
			break;
	if (i == 0 && x[0] == 0) {
		mfree(x);
		x = mint(0);
	} else {
		MLENGTH(x) = i + 1;
		MSIGN(x) = MSIGN(a);
	}
	*q = y;
	*r = x;
}

void mod(void);

void
eval_mod(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	mod();
}

void
mod(void)
{
	int n;
	save();
	p2 = pop();
	p1 = pop();
	if (iszero(p2))
		stop("mod function: divide by zero");
	if (!isnum(p1) || !isnum(p2)) {
		push_symbol(MOD);
		push(p1);
		push(p2);
		list(3);
		restore();
		return;
	}
	if (isdouble(p1)) {
		push(p1);
		n = pop_integer();
		if (n == (int) 0x80000000)
			stop("mod function: cannot convert float value to integer");
		push_integer(n);
		p1 = pop();
	}
	if (isdouble(p2)) {
		push(p2);
		n = pop_integer();
		if (n == (int) 0x80000000)
			stop("mod function: cannot convert float value to integer");
		push_integer(n);
		p2 = pop();
	}
	if (!isinteger(p1) || !isinteger(p2))
		stop("mod function: integer arguments expected");
	p3 = alloc();
	p3->k = NUM;
	p3->u.q.a = mmod(p1->u.q.a, p2->u.q.a);
	p3->u.q.b = mint(1);
	push(p3);
	restore();
}

// Bignum power

unsigned int *
mpow(unsigned int *a, unsigned int n)
{
	unsigned int *aa, *t;
	a = mcopy(a);
	aa = mint(1);
	for (;;) {
		if (n & 1) {
			t = mmul(aa, a);
			mfree(aa);
			aa = t;
		}
		n >>= 1;
		if (n == 0)
			break;
		t = mmul(a, a);
		mfree(a);
		a = t;
	}
	mfree(a);
	return aa;
}

// Bignum prime test (returns 1 if prime, 0 if not)

// Uses Algorithm P (probabilistic primality test) from p. 395 of
// "The Art of Computer Programming, Volume 2" by Donald E. Knuth.

int mprimef(unsigned int *, unsigned int *, int);

int
mprime(unsigned int *n)
{
	int i, k;
	unsigned int *q;
	// 1?
	if (MLENGTH(n) == 1 && n[0] == 1)
		return 0;
	// 2?
	if (MLENGTH(n) == 1 && n[0] == 2)
		return 1;
	// even?
	if ((n[0] & 1) == 0)
		return 0;
	// n = 1 + (2 ^ k) q
	q = mcopy(n);
	k = 0;
	do {
		mshiftright(q);
		k++;
	} while ((q[0] & 1) == 0);
	// try 25 times
	for (i = 0; i < 25; i++)
		if (mprimef(n, q, k) == 0)
			break;
	mfree(q);
	if (i < 25)
		return 0;
	else
		return 1;
}

//-----------------------------------------------------------------------------
//
//	This is the actual implementation of Algorithm P.
//
//	Input:		n		The number in question.
//
//			q		n = 1 + (2 ^ k) q
//
//			k
//
//	Output:		1		when n is probably prime
//
//			0		when n is definitely not prime
//
//-----------------------------------------------------------------------------

int
mprimef(unsigned int *n, unsigned int *q, int k)
{
	int i, j;
	unsigned int *t, *x, *y;
	// generate x
	t = mcopy(n);
	while (1) {
		for (i = 0; i < MLENGTH(t); i++)
			t[i] = rand();
		x = mmod(t, n);
		if (!MZERO(x) && !MEQUAL(x, 1))
			break;
		mfree(x);
	}
	mfree(t);
	// exponentiate
	y = mmodpow(x, q, n);
	// done?
	if (MEQUAL(y, 1)) {
		mfree(x);
		mfree(y);
		return 1;
	}
	j = 0;
	while (1) {
		// y = n - 1?
		t = msub(n, y);
		if (MEQUAL(t, 1)) {
			mfree(t);
			mfree(x);
			mfree(y);
			return 1;
		}
		mfree(t);
		if (++j == k) {
			mfree(x);
			mfree(y);
			return 0;
		}
		// y = (y ^ 2) mod n
		t = mmul(y, y);
		mfree(y);
		y = mmod(t, n);
		mfree(t);
		// y = 1?
		if (MEQUAL(y, 1)) {
			mfree(x);
			mfree(y);
			return 0;
		}
	}
}

//-----------------------------------------------------------------------------
//
//	Bignum root
//
//	Returns null pointer if not perfect root.
//
//	The sign of the radicand is ignored.
//
//-----------------------------------------------------------------------------

unsigned int *
mroot(unsigned int *n, unsigned int index)
{
	int i, j, k;
	unsigned int m, *x, *y;
	if (index == 0)
		stop("root index is zero");
	// count number of bits
	k = 32 * (MLENGTH(n) - 1);
	m = n[MLENGTH(n) - 1];
	while (m) {
		m >>= 1;
		k++;
	}
	if (k == 0)
		return mint(0);
	// initial guess
	k = (k - 1) / index;
	j = k / 32 + 1;
	x = mnew(j);
	MSIGN(x) = 1;
	MLENGTH(x) = j;
	for (i = 0; i < j; i++)
		x[i] = 0;
	while (k >= 0) {
		mp_set_bit(x, k);
		y = mpow(x, index);
		switch (mcmp(y, n)) {
		case -1:
			break;
		case 0:
			mfree(y);
			return x;
		case 1:
			mp_clr_bit(x, k);
			break;
		}
		mfree(y);
		k--;
	}
	mfree(x);
	return 0;
}

// bignum scanner

unsigned int *maddf(unsigned int *, int);
unsigned int *mmulf(unsigned int *, int);

unsigned int *
mscan(char *s)
{
	int sign;
	unsigned int *a, *b, *c;
	sign = 1;
	if (*s == '-') {
		sign = -1;
		s++;
	}
	a = mint(0);
	while (*s) {
		b = mmulf(a, 10);
		c = maddf(b, *s - '0');
		mfree(a);
		mfree(b);
		a = c;
		s++;
	}
	if (!MZERO(a))
		MSIGN(a) *= sign;
	return a;
}

unsigned int *
maddf(unsigned int *a, int n)
{
	unsigned int *b, *c;
	b = mint(n);
	c = madd(a, b);
	mfree(b);
	return c;
}

unsigned int *
mmulf(unsigned int *a, int n)
{
	unsigned int *b, *c;
	b = mint(n);
	c = mmul(a, b);
	mfree(b);
	return c;
}

// Convert bignum to string

char *
mstr(unsigned int *a)
{
	int k, n, r, sign;
	char c;
	static char *str;
	static int len;
	if (str == NULL) {
		str = (char *) malloc(1000);
		len = 1000;
	}
	// estimate string size
	n = 10 * MLENGTH(a) + 2;
	if (n > len) {
		free(str);
		str = (char *) malloc(n);
		len = n;
	}
	sign = MSIGN(a);
	a = mcopy(a);
	k = len - 1;
	str[k] = 0;
	for (;;) {
		k -= 9;
		r = divby1billion(a);
		c = str[k + 9];
		sprintf(str + k, "%09d", r);
		str[k + 9] = c;
		if (MZERO(a))
			break;
	}
	// remove leading zeroes
	while (str[k] == '0')
		k++;
	if (str[k] == 0)
		k--; // leave one leading zero
	// sign
	if (sign == -1) {
		k--;
		str[k] = '-';
	}
	mfree(a);
	return str + k;
}

// Returns remainder as function value, quotient returned in a.

int
divby1billion(unsigned int *a)
{
	int i;
	uint64_t kk = 0;
	for (i = MLENGTH(a) - 1; i >= 0; i--) {
		kk = kk << 32 | a[i];
		a[i] = (uint32_t) (kk / 1000000000); // compiler warns w/o cast
		kk -= (uint64_t) 1000000000 * a[i];
	}
	// length of quotient
	for (i = MLENGTH(a) - 1; i > 0; i--)
		if (a[i])
			break;
	MLENGTH(a) = i + 1;
	return (int) kk; // compiler warns w/o cast
}

// Symbolic multiplication

void
multiply(void)
{
	check_esc_flag();
	if (isnum(stack[tos - 2]) && isnum(stack[tos - 1]))
		multiply_numbers();
	else {
		save();
		yymultiply();
		restore();
	}
}

void
yymultiply(void)
{
	int h, i, n;
	// pop operands
	p2 = pop();
	p1 = pop();
	h = tos;
	// is either operand zero?
	if (p1->k != TENSOR && p2->k != TENSOR && (iszero(p1) || iszero(p2))) {
		push(zero);
		return;
	}
	// is either operand a sum?
	if (expanding && isadd(p1)) {
		p1 = cdr(p1);
		push(zero);
		while (iscons(p1)) {
			push(car(p1));
			push(p2);
			multiply();
			add();
			p1 = cdr(p1);
		}
		return;
	}
	if (expanding && isadd(p2)) {
		p2 = cdr(p2);
		push(zero);
		while (iscons(p2)) {
			push(p1);
			push(car(p2));
			multiply();
			add();
			p2 = cdr(p2);
		}
		return;
	}
	// scalar times tensor?
	if (!istensor(p1) && istensor(p2)) {
		push(p1);
		push(p2);
		scalar_times_tensor();
		return;
	}
	// tensor times scalar?
	if (istensor(p1) && !istensor(p2)) {
		push(p1);
		push(p2);
		tensor_times_scalar();
		return;
	}
	// adjust operands
	if (car(p1) == symbol(MULTIPLY))
		p1 = cdr(p1);
	else {
		push(p1);
		list(1);
		p1 = pop();
	}
	if (car(p2) == symbol(MULTIPLY))
		p2 = cdr(p2);
	else {
		push(p2);
		list(1);
		p2 = pop();
	}
	// handle numerical coefficients
	if (isnum(car(p1)) && isnum(car(p2))) {
		push(car(p1));
		push(car(p2));
		multiply_numbers();
		p1 = cdr(p1);
		p2 = cdr(p2);
	} else if (isnum(car(p1))) {
		push(car(p1));
		p1 = cdr(p1);
	} else if (isnum(car(p2))) {
		push(car(p2));
		p2 = cdr(p2);
	} else
		push(one);
	parse_p1();
	parse_p2();
	while (iscons(p1) && iscons(p2)) {
		if (caar(p1) == symbol(OPERATOR) && caar(p2) == symbol(OPERATOR)) {
			push_symbol(OPERATOR);
			push(cdar(p1));
			push(cdar(p2));
			append();
			cons();
			p1 = cdr(p1);
			p2 = cdr(p2);
			parse_p1();
			parse_p2();
			continue;
		}
		switch (cmp_expr(p3, p4)) {
		case -1:
			push(car(p1));
			p1 = cdr(p1);
			parse_p1();
			break;
		case 1:
			push(car(p2));
			p2 = cdr(p2);
			parse_p2();
			break;
		case 0:
			combine_factors(h);
			p1 = cdr(p1);
			p2 = cdr(p2);
			parse_p1();
			parse_p2();
			break;
		default:
			stop("internal error 2");
			break;
		}
	}
	// push remaining factors, if any
	while (iscons(p1)) {
		push(car(p1));
		p1 = cdr(p1);
	}
	while (iscons(p2)) {
		push(car(p2));
		p2 = cdr(p2);
	}
	// normalize radical factors
	// example: 2*2(-1/2) -> 2^(1/2)
	// must be done after merge because merge may produce radical
	// example: 2^(1/2-a)*2^a -> 2^(1/2)
	normalize_radical_factors(h);
	// this hack should not be necessary, unless power returns a multiply
	//for (i = h; i < tos; i++) {
	//	if (car(stack[i]) == symbol(MULTIPLY)) {
	//		multiply_all(tos - h);
	//		return;
	//	}
	//}
	if (expanding) {
		for (i = h; i < tos; i++) {
			if (isadd(stack[i])) {
				multiply_all(tos - h);
				return;
			}
		}
	}
	// n is the number of result factors on the stack
	n = tos - h;
	if (n == 1)
		return;
	// discard integer 1
	if (isrational(stack[h]) && equaln(stack[h], 1)) {
		if (n == 2) {
			p7 = pop();
			pop();
			push(p7);
		} else {
			stack[h] = symbol(MULTIPLY);
			list(n);
		}
		return;
	}
	list(n);
	p7 = pop();
	push_symbol(MULTIPLY);
	push(p7);
	cons();
}

// Decompose a factor into base and power.
//
// input:	car(p1)		factor
//
// output:	p3		factor's base
//
//		p5		factor's power (possibly 1)

void
parse_p1(void)
{
	p3 = car(p1);
	p5 = one;
	if (car(p3) == symbol(POWER)) {
		p5 = caddr(p3);
		p3 = cadr(p3);
	}
}

// Decompose a factor into base and power.
//
// input:	car(p2)		factor
//
// output:	p4		factor's base
//
//		p6		factor's power (possibly 1)

void
parse_p2(void)
{
	p4 = car(p2);
	p6 = one;
	if (car(p4) == symbol(POWER)) {
		p6 = caddr(p4);
		p4 = cadr(p4);
	}
}

void
combine_factors(int h)
{
	push(p4);
	push(p5);
	push(p6);
	add();
	power();
	p7 = pop();
	if (isnum(p7)) {
		push(stack[h]);
		push(p7);
		multiply_numbers();
		stack[h] = pop();
	} else if (car(p7) == symbol(MULTIPLY)) {
		// power can return number * factor (i.e. -1 * i)
		if (isnum(cadr(p7)) && cdddr(p7) == symbol(NIL)) {
			push(stack[h]);
			push(cadr(p7));
			multiply_numbers();
			stack[h] = pop();
			push(caddr(p7));
		} else
			push(p7);
	} else
		push(p7);
}

void
multiply_noexpand(void)
{
	int x;
	x = expanding;
	expanding = 0;
	multiply();
	expanding = x;
}

// multiply n factors on stack

void
multiply_all(int n)
{
	int h, i;
	if (n == 1)
		return;
	if (n == 0) {
		push(one);
		return;
	}
	h = tos - n;
	push(stack[h]);
	for (i = 1; i < n; i++) {
		push(stack[h + i]);
		multiply();
	}
	stack[h] = pop();
	tos = h + 1;
}

void
multiply_all_noexpand(int n)
{
	int x;
	x = expanding;
	expanding = 0;
	multiply_all(n);
	expanding = x;
}

//-----------------------------------------------------------------------------
//
//	Symbolic division
//
//	Input:		Dividend and divisor on stack
//
//	Output:		Quotient on stack
//
//-----------------------------------------------------------------------------

void
divide(void)
{
	if (isnum(stack[tos - 2]) && isnum(stack[tos - 1]))
		divide_numbers();
	else {
		inverse();
		multiply();
	}
}

void
inverse(void)
{
	if (isnum(stack[tos - 1]))
		invert_number();
	else {
		push_integer(-1);
		power();
	}
}

void
reciprocate(void)
{
	if (isnum(stack[tos - 1]))
		invert_number();
	else {
		push_integer(-1);
		power();
	}
}

void
negate(void)
{
	if (isnum(stack[tos - 1]))
		negate_number();
	else {
		push_integer(-1);
		multiply();
	}
}

void
negate_expand(void)
{
	int x;
	x = expanding;
	expanding = 1;
	negate();
	expanding = x;
}

void
negate_noexpand(void)
{
	int x;
	x = expanding;
	expanding = 0;
	negate();
	expanding = x;
}

//-----------------------------------------------------------------------------
//
//	Normalize radical factors
//
//	Input:		stack[h]	Coefficient factor, possibly 1
//
//			stack[h + 1]	Second factor
//
//			stack[tos - 1]	Last factor
//
//	Output:		Reduced coefficent and normalized radicals (maybe)
//
//	Example:	2*2^(-1/2) -> 2^(1/2)
//
//	(power number number) is guaranteed to have the following properties:
//
//	1. Base is an integer
//
//	2. Absolute value of exponent < 1
//
//	These properties are assured by the power function.
//
//-----------------------------------------------------------------------------

#undef A
#undef B
#undef BASE
#undef EXPO
#undef TMP

#define A p1
#define B p2
#define BASE p3
#define EXPO p4
#define TMP p5

void
normalize_radical_factors(int h)
{
	int i;
	// if coeff is 1 or floating then don't bother
	if (isplusone(stack[h]) || isminusone(stack[h]) || isdouble(stack[h]))
		return;
	// if no radicals then don't bother
	for (i = h + 1; i < tos; i++)
		if (is_radical_number(stack[i]))
			break;
	if (i == tos)
		return;
	// ok, try to simplify
	save();
	// numerator
	push(stack[h]);
	mp_numerator();
	A = pop();
	for (i = h + 1; i < tos; i++) {
		if (isplusone(A) || isminusone(A))
			break;
		if (!is_radical_number(stack[i]))
			continue;
		BASE = cadr(stack[i]);
		EXPO = caddr(stack[i]);
		// exponent must be negative
		if (!isnegativenumber(EXPO))
			continue;
		// numerator divisible by BASE?
		push(A);
		push(BASE);
		divide();
		TMP = pop();
		if (!isinteger(TMP))
			continue;
		// reduce numerator
		A = TMP;
		// invert radical
		push_symbol(POWER);
		push(BASE);
		push(one);
		push(EXPO);
		add();
		list(3);
		stack[i] = pop();
	}
	// denominator
	push(stack[h]);
	mp_denominator();
	B = pop();
	for (i = h + 1; i < tos; i++) {
		if (isplusone(B))
			break;
		if (!is_radical_number(stack[i]))
			continue;
		BASE = cadr(stack[i]);
		EXPO = caddr(stack[i]);
		// exponent must be positive
		if (isnegativenumber(EXPO))
			continue;
		// denominator divisible by BASE?
		push(B);
		push(BASE);
		divide();
		TMP = pop();
		if (!isinteger(TMP))
			continue;
		// reduce denominator
		B = TMP;
		// invert radical
		push_symbol(POWER);
		push(BASE);
		push(EXPO);
		push(one);
		subtract();
		list(3);
		stack[i] = pop();
	}
	// reconstitute the coefficient
	push(A);
	push(B);
	divide();
	stack[h] = pop();
	restore();
}

// don't include i

int
is_radical_number(U *p)
{
	// don't use i
	if (car(p) == symbol(POWER) && isnum(cadr(p)) && isnum(caddr(p)) && !isminusone(cadr(p)))
		return 1;
	else
		return 0;
}

//-----------------------------------------------------------------------------
//
//	> a*hilbert(2)
//	((a,1/2*a),(1/2*a,1/3*a))
//
//	Note that "a" is presumed to be a scalar. Is this correct?
//
//	Yes, because "*" has no meaning if "a" is a tensor.
//	To multiply tensors, "dot" or "outer" should be used.
//
//	> dot(a,hilbert(2))
//	dot(a,((1,1/2),(1/2,1/3)))
//
//	In this case "a" could be a scalar or tensor so the result is not
//	expanded.
//
//-----------------------------------------------------------------------------

// find the roots of a polynomial numerically

#undef YMAX

#define YMAX 101
#define DELTA 1.0e-6
#define EPSILON 1.0e-9
#define YABS(z) sqrt((z).r * (z).r + (z).i * (z).i)
#define RANDOM (4.0 * (double) rand() / (double) RAND_MAX - 2.0)

struct {
	double r, i;
} a, b, x, y, fa, fb, dx, df, c[YMAX];

void
eval_nroots(void)
{
	int h, i, k, n;
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		guess();
	else
		push(p2);
	p2 = pop();
	p1 = pop();
	if (!ispoly(p1, p2))
		stop("nroots: polynomial?");
	// mark the stack
	h = tos;
	// get the coefficients
	push(p1);
	push(p2);
	n = coeff();
	if (n > YMAX)
		stop("nroots: degree?");
	// convert the coefficients to real and imaginary doubles
	for (i = 0; i < n; i++) {
		push(stack[h + i]);
		real();
		yyfloat();
		eval();
		p1 = pop();
		push(stack[h + i]);
		imag();
		yyfloat();
		eval();
		p2 = pop();
		if (!isdouble(p1) || !isdouble(p2))
			stop("nroots: coefficients?");
		c[i].r = p1->u.d;
		c[i].i = p2->u.d;
	}
	// pop the coefficients
	tos = h;
	// n is the number of coefficients, n = deg(p) + 1
	monic(n);
	for (k = n; k > 1; k--) {
		findroot(k);
		if (fabs(a.r) < DELTA)
			a.r = 0.0;
		if (fabs(a.i) < DELTA)
			a.i = 0.0;
		push_double(a.r);
		push_double(a.i);
		push(imaginaryunit);
		multiply();
		add();
		divpoly_FIXME(k);
	}
	// now make n equal to the number of roots
	n = tos - h;
	if (n > 1) {
		sort_stack(n);
		p1 = alloc_tensor(n);
		p1->u.tensor->ndim = 1;
		p1->u.tensor->dim[0] = n;
		for (i = 0; i < n; i++)
			p1->u.tensor->elem[i] = stack[h + i];
		tos = h;
		push(p1);
	}
}

// divide the polynomial by its leading coefficient

void
monic(int n)
{
	int k;
	double t;
	y = c[n - 1];
	t = y.r * y.r + y.i * y.i;
	for (k = 0; k < n - 1; k++) {
		c[k].r = (c[k].r * y.r + c[k].i * y.i) / t;
		c[k].i = (c[k].i * y.r - c[k].r * y.i) / t;
	}
	c[n - 1].r = 1.0;
	c[n - 1].i = 0.0;
}

// uses the secant method

void
findroot(int n)
{
	int j, k;
	double t;
	if (YABS(c[0]) < DELTA) {
		a.r = 0.0;
		a.i = 0.0;
		return;
	}
	for (j = 0; j < 100; j++) {
		a.r = RANDOM;
		a.i = RANDOM;
		compute_fa(n);
		b = a;
		fb = fa;
		a.r = RANDOM;
		a.i = RANDOM;
		for (k = 0; k < 1000; k++) {
			compute_fa(n);
			if (YABS(fa) < EPSILON)
				return;
			if (YABS(fa) < YABS(fb)) {
				x = a;
				a = b;
				b = x;
				x = fa;
				fa = fb;
				fb = x;
			}
			// dx = b - a
			dx.r = b.r - a.r;
			dx.i = b.i - a.i;
			// df = fb - fa
			df.r = fb.r - fa.r;
			df.i = fb.i - fa.i;
			// y = dx / df
			t = df.r * df.r + df.i * df.i;
			if (t == 0.0)
				break;
			y.r = (dx.r * df.r + dx.i * df.i) / t;
			y.i = (dx.i * df.r - dx.r * df.i) / t;
			// a = b - y * fb
			a.r = b.r - (y.r * fb.r - y.i * fb.i);
			a.i = b.i - (y.r * fb.i + y.i * fb.r);
		}
	}
	stop("nroots: convergence error");
}

void
compute_fa(int n)
{
	int k;
	double t;
	// x = a
	x.r = a.r;
	x.i = a.i;
	// fa = c0 + c1 * x
	fa.r = c[0].r + c[1].r * x.r - c[1].i * x.i;
	fa.i = c[0].i + c[1].r * x.i + c[1].i * x.r;
	for (k = 2; k < n; k++) {
		// x = a * x
		t = a.r * x.r - a.i * x.i;
		x.i = a.r * x.i + a.i * x.r;
		x.r = t;
		// fa += c[k] * x
		fa.r += c[k].r * x.r - c[k].i * x.i;
		fa.i += c[k].r * x.i + c[k].i * x.r;
	}
}

// divide the polynomial by x - a

void
divpoly_FIXME(int n)
{
	int k;
	for (k = n - 1; k > 0; k--) {
		c[k - 1].r += c[k].r * a.r - c[k].i * a.i;
		c[k - 1].i += c[k].i * a.r + c[k].r * a.i;
	}
	if (YABS(c[0]) > DELTA)
		stop("nroots: residual error");
	for (k = 0; k < n - 1; k++) {
		c[k].r = c[k + 1].r;
		c[k].i = c[k + 1].i;
	}
}

void
eval_numerator(void)
{
	push(cadr(p1));
	eval();
	numerator();
}

void
numerator(void)
{
	int h;
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD)) {
		push(p1);
		rationalize();
		p1 = pop();
	}
	if (car(p1) == symbol(MULTIPLY)) {
		h = tos;
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			numerator();
			p1 = cdr(p1);
		}
		multiply_all(tos - h);
	} else if (isrational(p1)) {
		push(p1);
		mp_numerator();
	} else if (car(p1) == symbol(POWER) && isnegativeterm(caddr(p1)))
		push(one);
	else
		push(p1);
	restore();
}

// Outer product of tensors

void
eval_outer(void)
{
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval();
		outer();
		p1 = cdr(p1);
	}
}

void
outer(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (istensor(p1) && istensor(p2))
		yyouter();
	else {
		push(p1);
		push(p2);
		if (istensor(p1))
			tensor_times_scalar();
		else if (istensor(p2))
			scalar_times_tensor();
		else
			multiply();
	}
	restore();
}

void
yyouter(void)
{
	int i, j, k, ndim, nelem;
	ndim = p1->u.tensor->ndim + p2->u.tensor->ndim;
	if (ndim > MAXDIM)
		stop("outer: rank of result exceeds maximum");
	nelem = p1->u.tensor->nelem * p2->u.tensor->nelem;
	p3 = alloc_tensor(nelem);
	p3->u.tensor->ndim = ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	j = i;
	for (i = 0; i < p2->u.tensor->ndim; i++)
		p3->u.tensor->dim[j + i] = p2->u.tensor->dim[i];
	k = 0;
	for (i = 0; i < p1->u.tensor->nelem; i++)
		for (j = 0; j < p2->u.tensor->nelem; j++) {
			push(p1->u.tensor->elem[i]);
			push(p2->u.tensor->elem[j]);
			multiply();
			p3->u.tensor->elem[k++] = pop();
		}
	push(p3);
}

/* Partition a term

	Input stack:

		term (factor or product of factors)

		free variable

	Output stack:

		constant expression

		variable expression
*/

void
partition(void)
{
	save();
	p2 = pop();
	p1 = pop();
	push_integer(1);
	p3 = pop();
	p4 = p3;
	p1 = cdr(p1);
	while (iscons(p1)) {
		if (find(car(p1), p2)) {
			push(p4);
			push(car(p1));
			multiply();
			p4 = pop();
		} else {
			push(p3);
			push(car(p1));
			multiply();
			p3 = pop();
		}
		p1 = cdr(p1);
	}
	push(p3);
	push(p4);
	restore();
}

/* Convert complex z to polar form

	Input:		push	z

	Output:		Result on stack

	polar(z) = mag(z) * exp(i * arg(z))
*/

void
eval_polar(void)
{
	push(cadr(p1));
	eval();
	polar();
}

void
polar(void)
{
	save();
	p1 = pop();
	push(p1);
	mag();
	push(imaginaryunit);
	push(p1);
	arg();
	multiply();
	exponential();
	multiply();
	restore();
}

// Factor using the Pollard rho method

unsigned int *global_n;

void
factor_number(void)
{
	int h;
	save();
	p1 = pop();
	// 0 or 1?
	if (equaln(p1, 0) || equaln(p1, 1) || equaln(p1, -1)) {
		push(p1);
		restore();
		return;
	}
	global_n = mcopy(p1->u.q.a);
	h = tos;
	factor_a();
	if (tos - h > 1) {
		list(tos - h);
		push_symbol(MULTIPLY);
		swap();
		cons();
	}
	restore();
}

// factor using table look-up, then switch to rho method if necessary

// From TAOCP Vol. 2 by Knuth, p. 380 (Algorithm A)

void
factor_a(void)
{
	int k;
	if (MSIGN(global_n) == -1) {
		MSIGN(global_n) = 1;
		push_integer(-1);
	}
	for (k = 0; k < 10000; k++) {
		try_kth_prime(k);
		// if n is 1 then we're done
		if (MLENGTH(global_n) == 1 && global_n[0] == 1) {
			mfree(global_n);
			return;
		}
	}
	factor_b();
}

void
try_kth_prime(int k)
{
	int count;
	unsigned int *d, *q, *r;
	d = mint(primetab[k]);
	count = 0;
	while (1) {
		// if n is 1 then we're done
		if (MLENGTH(global_n) == 1 && global_n[0] == 1) {
			if (count)
				push_factor(d, count);
			else
				mfree(d);
			return;
		}
		mdivrem(&q, &r, global_n, d);
		// continue looping while remainder is zero
		if (MLENGTH(r) == 1 && r[0] == 0) {
			count++;
			mfree(r);
			mfree(global_n);
			global_n = q;
		} else {
			mfree(r);
			break;
		}
	}
	if (count)
		push_factor(d, count);
	// q = n/d, hence if q < d then n < d^2 so n is prime
	if (mcmp(q, d) == -1) {
		push_factor(global_n, 1);
		global_n = mint(1);
	}
	if (count == 0)
		mfree(d);
	mfree(q);
}

// From TAOCP Vol. 2 by Knuth, p. 385 (Algorithm B)

int
factor_b(void)
{
	int k, l;
	unsigned int *g, *k1, *t, *x, *xprime;
	k1 = mint(1);
	x = mint(5);
	xprime = mint(2);
	k = 1;
	l = 1;
	while (1) {
		if (mprime(global_n)) {
			push_factor(global_n, 1);
			mfree(k1);
			mfree(x);
			mfree(xprime);
			return 0;
		}
		while (1) {
			if (esc_flag) {
				mfree(k1);
				mfree(global_n);
				mfree(x);
				mfree(xprime);
				stop(NULL);
			}
			// g = gcd(x' - x, n)
			t = msub(xprime, x);
			MSIGN(t) = 1;
			g = mgcd(t, global_n);
			mfree(t);
			if (MEQUAL(g, 1)) {
				mfree(g);
				if (--k == 0) {
					mfree(xprime);
					xprime = mcopy(x);
					l *= 2;
					k = l;
				}
				// x = (x ^ 2 + 1) mod n
				t = mmul(x, x);
				mfree(x);
				x = madd(t, k1);
				mfree(t);
				t = mmod(x, global_n);
				mfree(x);
				x = t;
				continue;
			}
			push_factor(g, 1);
			if (mcmp(g, global_n) == 0) {
				mfree(k1);
				mfree(global_n);
				mfree(x);
				mfree(xprime);
				return -1;
			}
			// n = n / g
			t = mdiv(global_n, g);
			mfree(global_n);
			global_n = t;
			// x = x mod n
			t = mmod(x, global_n);
			mfree(x);
			x = t;
			// xprime = xprime mod n
			t = mmod(xprime, global_n);
			mfree(xprime);
			xprime = t;
			break;
		}
	}
}

void
push_factor(unsigned int *d, int count)
{
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = d;
	p1->u.q.b = mint(1);
	push(p1);
	if (count > 1) {
		push_symbol(POWER);
		swap();
		p1 = alloc();
		p1->k = NUM;
		p1->u.q.a = mint(count);
		p1->u.q.b = mint(1);
		push(p1);
		list(3);
	}
}

/* Power function

	Input:		push	Base

			push	Exponent

	Output:		Result on stack
*/

void
eval_power(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	power();
}

void
power(void)
{
	save();
	yypower();
	restore();
}

void
yypower(void)
{
	int i, n;
	p2 = pop();
	p1 = pop();
	// both base and exponent are rational numbers?
	if (isrational(p1) && isrational(p2)) {
		push(p1);
		push(p2);
		qpow();
		return;
	}
	// both base and exponent are either rational or double?
	if (isnum(p1) && isnum(p2)) {
		push(p1);
		push(p2);
		dpow();
		return;
	}
	if (istensor(p1)) {
		power_tensor();
		return;
	}
	if (p1 == symbol(EXP1) && car(p2) == symbol(LOG)) {
		push(cadr(p2));
		return;
	}
	if (p1 == symbol(EXP1) && isdouble(p2)) {
		push_double(exp(p2->u.d));
		return;
	}
	//	1 ^ a		->	1
	//	a ^ 0		->	1
	if (equal(p1, one) || iszero(p2)) {
		push(one);
		return;
	}
	//	a ^ 1		->	a
	if (equal(p2, one)) {
		push(p1);
		return;
	}
	//	(a * b) ^ c	->	(a ^ c) * (b ^ c)
	if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		push(car(p1));
		push(p2);
		power();
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			push(p2);
			power();
			multiply();
			p1 = cdr(p1);
		}
		return;
	}
	//	(a ^ b) ^ c	->	a ^ (b * c)
	if (car(p1) == symbol(POWER)) {
		push(cadr(p1));
		push(caddr(p1));
		push(p2);
		multiply();
		power();
		return;
	}
	//	(a + b) ^ n	->	(a + b) * (a + b) ...
	if (expanding && isadd(p1) && isnum(p2)) {
		push(p2);
		n = pop_integer();
		if (n > 0) {
			push(p1);
			for (i = 1; i < n; i++) {
				push(p1);
				multiply();
			}
			return;
		}
	}
	//	sin(x) ^ 2n -> (1 - cos(x) ^ 2) ^ n
	if (trigmode == 1 && car(p1) == symbol(SIN) && iseveninteger(p2)) {
		push_integer(1);
		push(cadr(p1));
		cosine();
		push_integer(2);
		power();
		subtract();
		push(p2);
		push_rational(1, 2);
		multiply();
		power();
		return;
	}
	//	cos(x) ^ 2n -> (1 - sin(x) ^ 2) ^ n
	if (trigmode == 2 && car(p1) == symbol(COS) && iseveninteger(p2)) {
		push_integer(1);
		push(cadr(p1));
		sine();
		push_integer(2);
		power();
		subtract();
		push(p2);
		push_rational(1, 2);
		multiply();
		power();
		return;
	}
	// complex number? (just number, not expression)
	if (iscomplexnumber(p1)) {
		// integer power?
		// n will be negative here, positive n already handled
		if (isinteger(p2)) {
			//               /        \  n
			//         -n   |  a - ib  |
			// (a + ib)   = | -------- |
			//              |   2   2  |
			//               \ a + b  /
			push(p1);
			conjugate();
			p3 = pop();
			push(p3);
			push(p3);
			push(p1);
			multiply();
			divide();
			push(p2);
			negate();
			power();
			return;
		}
		// noninteger or floating power?
		if (isnum(p2)) {
#if 1			// use polar form
			push(p1);
			mag();
			push(p2);
			power();
			push_integer(-1);
			push(p1);
			arg();
			push(p2);
			multiply();
			push_symbol(PI);
			divide();
			power();
			multiply();
#else			// use exponential form
			push(p1);
			mag();
			push(p2);
			power();
			push_symbol(E);
			push(p1);
			arg();
			push(p2);
			multiply();
			push(imaginaryunit);
			multiply();
			power();
			multiply();
#endif
			return;
		}
	}
	if (simplify_polar())
		return;
	push_symbol(POWER);
	push(p1);
	push(p2);
	list(3);
}

// exp(n/2 i pi) ?

// p2 is the exponent expression

// clobbers p3

int
simplify_polar(void)
{
	int n;
	n = isquarterturn(p2);
	switch(n) {
	case 0:
		break;
	case 1:
		push_integer(1);
		return 1;
	case 2:
		push_integer(-1);
		return 1;
	case 3:
		push(imaginaryunit);
		return 1;
	case 4:
		push(imaginaryunit);
		negate();
		return 1;
	}
	if (car(p2) == symbol(ADD)) {
		p3 = cdr(p2);
		while (iscons(p3)) {
			n = isquarterturn(car(p3));
			if (n)
				break;
			p3 = cdr(p3);
		}
		switch (n) {
		case 0:
			return 0;
		case 1:
			push_integer(1);
			break;
		case 2:
			push_integer(-1);
			break;
		case 3:
			push(imaginaryunit);
			break;
		case 4:
			push(imaginaryunit);
			negate();
			break;
		}
		push(p2);
		push(car(p3));
		subtract();
		exponential();
		multiply();
		return 1;
	}
	return 0;
}

//-----------------------------------------------------------------------------
//
//	Look up the nth prime
//
//	Input:		n on stack (0 < n < 10001)
//
//	Output:		nth prime on stack
//
//-----------------------------------------------------------------------------

void
eval_prime(void)
{
	push(cadr(p1));
	eval();
	prime();
}

void
prime(void)
{
	int n;
	n = pop_integer();
	if (n < 1 || n > MAXPRIMETAB)
		stop("prime: Argument out of range.");
	n = primetab[n - 1];
	push_integer(n);
}

int primetab[10000] = {
2,3,5,7,11,13,17,19,
23,29,31,37,41,43,47,53,
59,61,67,71,73,79,83,89,
97,101,103,107,109,113,127,131,
137,139,149,151,157,163,167,173,
179,181,191,193,197,199,211,223,
227,229,233,239,241,251,257,263,
269,271,277,281,283,293,307,311,
313,317,331,337,347,349,353,359,
367,373,379,383,389,397,401,409,
419,421,431,433,439,443,449,457,
461,463,467,479,487,491,499,503,
509,521,523,541,547,557,563,569,
571,577,587,593,599,601,607,613,
617,619,631,641,643,647,653,659,
661,673,677,683,691,701,709,719,
727,733,739,743,751,757,761,769,
773,787,797,809,811,821,823,827,
829,839,853,857,859,863,877,881,
883,887,907,911,919,929,937,941,
947,953,967,971,977,983,991,997,
1009,1013,1019,1021,1031,1033,1039,1049,
1051,1061,1063,1069,1087,1091,1093,1097,
1103,1109,1117,1123,1129,1151,1153,1163,
1171,1181,1187,1193,1201,1213,1217,1223,
1229,1231,1237,1249,1259,1277,1279,1283,
1289,1291,1297,1301,1303,1307,1319,1321,
1327,1361,1367,1373,1381,1399,1409,1423,
1427,1429,1433,1439,1447,1451,1453,1459,
1471,1481,1483,1487,1489,1493,1499,1511,
1523,1531,1543,1549,1553,1559,1567,1571,
1579,1583,1597,1601,1607,1609,1613,1619,
1621,1627,1637,1657,1663,1667,1669,1693,
1697,1699,1709,1721,1723,1733,1741,1747,
1753,1759,1777,1783,1787,1789,1801,1811,
1823,1831,1847,1861,1867,1871,1873,1877,
1879,1889,1901,1907,1913,1931,1933,1949,
1951,1973,1979,1987,1993,1997,1999,2003,
2011,2017,2027,2029,2039,2053,2063,2069,
2081,2083,2087,2089,2099,2111,2113,2129,
2131,2137,2141,2143,2153,2161,2179,2203,
2207,2213,2221,2237,2239,2243,2251,2267,
2269,2273,2281,2287,2293,2297,2309,2311,
2333,2339,2341,2347,2351,2357,2371,2377,
2381,2383,2389,2393,2399,2411,2417,2423,
2437,2441,2447,2459,2467,2473,2477,2503,
2521,2531,2539,2543,2549,2551,2557,2579,
2591,2593,2609,2617,2621,2633,2647,2657,
2659,2663,2671,2677,2683,2687,2689,2693,
2699,2707,2711,2713,2719,2729,2731,2741,
2749,2753,2767,2777,2789,2791,2797,2801,
2803,2819,2833,2837,2843,2851,2857,2861,
2879,2887,2897,2903,2909,2917,2927,2939,
2953,2957,2963,2969,2971,2999,3001,3011,
3019,3023,3037,3041,3049,3061,3067,3079,
3083,3089,3109,3119,3121,3137,3163,3167,
3169,3181,3187,3191,3203,3209,3217,3221,
3229,3251,3253,3257,3259,3271,3299,3301,
3307,3313,3319,3323,3329,3331,3343,3347,
3359,3361,3371,3373,3389,3391,3407,3413,
3433,3449,3457,3461,3463,3467,3469,3491,
3499,3511,3517,3527,3529,3533,3539,3541,
3547,3557,3559,3571,3581,3583,3593,3607,
3613,3617,3623,3631,3637,3643,3659,3671,
3673,3677,3691,3697,3701,3709,3719,3727,
3733,3739,3761,3767,3769,3779,3793,3797,
3803,3821,3823,3833,3847,3851,3853,3863,
3877,3881,3889,3907,3911,3917,3919,3923,
3929,3931,3943,3947,3967,3989,4001,4003,
4007,4013,4019,4021,4027,4049,4051,4057,
4073,4079,4091,4093,4099,4111,4127,4129,
4133,4139,4153,4157,4159,4177,4201,4211,
4217,4219,4229,4231,4241,4243,4253,4259,
4261,4271,4273,4283,4289,4297,4327,4337,
4339,4349,4357,4363,4373,4391,4397,4409,
4421,4423,4441,4447,4451,4457,4463,4481,
4483,4493,4507,4513,4517,4519,4523,4547,
4549,4561,4567,4583,4591,4597,4603,4621,
4637,4639,4643,4649,4651,4657,4663,4673,
4679,4691,4703,4721,4723,4729,4733,4751,
4759,4783,4787,4789,4793,4799,4801,4813,
4817,4831,4861,4871,4877,4889,4903,4909,
4919,4931,4933,4937,4943,4951,4957,4967,
4969,4973,4987,4993,4999,5003,5009,5011,
5021,5023,5039,5051,5059,5077,5081,5087,
5099,5101,5107,5113,5119,5147,5153,5167,
5171,5179,5189,5197,5209,5227,5231,5233,
5237,5261,5273,5279,5281,5297,5303,5309,
5323,5333,5347,5351,5381,5387,5393,5399,
5407,5413,5417,5419,5431,5437,5441,5443,
5449,5471,5477,5479,5483,5501,5503,5507,
5519,5521,5527,5531,5557,5563,5569,5573,
5581,5591,5623,5639,5641,5647,5651,5653,
5657,5659,5669,5683,5689,5693,5701,5711,
5717,5737,5741,5743,5749,5779,5783,5791,
5801,5807,5813,5821,5827,5839,5843,5849,
5851,5857,5861,5867,5869,5879,5881,5897,
5903,5923,5927,5939,5953,5981,5987,6007,
6011,6029,6037,6043,6047,6053,6067,6073,
6079,6089,6091,6101,6113,6121,6131,6133,
6143,6151,6163,6173,6197,6199,6203,6211,
6217,6221,6229,6247,6257,6263,6269,6271,
6277,6287,6299,6301,6311,6317,6323,6329,
6337,6343,6353,6359,6361,6367,6373,6379,
6389,6397,6421,6427,6449,6451,6469,6473,
6481,6491,6521,6529,6547,6551,6553,6563,
6569,6571,6577,6581,6599,6607,6619,6637,
6653,6659,6661,6673,6679,6689,6691,6701,
6703,6709,6719,6733,6737,6761,6763,6779,
6781,6791,6793,6803,6823,6827,6829,6833,
6841,6857,6863,6869,6871,6883,6899,6907,
6911,6917,6947,6949,6959,6961,6967,6971,
6977,6983,6991,6997,7001,7013,7019,7027,
7039,7043,7057,7069,7079,7103,7109,7121,
7127,7129,7151,7159,7177,7187,7193,7207,
7211,7213,7219,7229,7237,7243,7247,7253,
7283,7297,7307,7309,7321,7331,7333,7349,
7351,7369,7393,7411,7417,7433,7451,7457,
7459,7477,7481,7487,7489,7499,7507,7517,
7523,7529,7537,7541,7547,7549,7559,7561,
7573,7577,7583,7589,7591,7603,7607,7621,
7639,7643,7649,7669,7673,7681,7687,7691,
7699,7703,7717,7723,7727,7741,7753,7757,
7759,7789,7793,7817,7823,7829,7841,7853,
7867,7873,7877,7879,7883,7901,7907,7919,
7927,7933,7937,7949,7951,7963,7993,8009,
8011,8017,8039,8053,8059,8069,8081,8087,
8089,8093,8101,8111,8117,8123,8147,8161,
8167,8171,8179,8191,8209,8219,8221,8231,
8233,8237,8243,8263,8269,8273,8287,8291,
8293,8297,8311,8317,8329,8353,8363,8369,
8377,8387,8389,8419,8423,8429,8431,8443,
8447,8461,8467,8501,8513,8521,8527,8537,
8539,8543,8563,8573,8581,8597,8599,8609,
8623,8627,8629,8641,8647,8663,8669,8677,
8681,8689,8693,8699,8707,8713,8719,8731,
8737,8741,8747,8753,8761,8779,8783,8803,
8807,8819,8821,8831,8837,8839,8849,8861,
8863,8867,8887,8893,8923,8929,8933,8941,
8951,8963,8969,8971,8999,9001,9007,9011,
9013,9029,9041,9043,9049,9059,9067,9091,
9103,9109,9127,9133,9137,9151,9157,9161,
9173,9181,9187,9199,9203,9209,9221,9227,
9239,9241,9257,9277,9281,9283,9293,9311,
9319,9323,9337,9341,9343,9349,9371,9377,
9391,9397,9403,9413,9419,9421,9431,9433,
9437,9439,9461,9463,9467,9473,9479,9491,
9497,9511,9521,9533,9539,9547,9551,9587,
9601,9613,9619,9623,9629,9631,9643,9649,
9661,9677,9679,9689,9697,9719,9721,9733,
9739,9743,9749,9767,9769,9781,9787,9791,
9803,9811,9817,9829,9833,9839,9851,9857,
9859,9871,9883,9887,9901,9907,9923,9929,
9931,9941,9949,9967,9973,10007,10009,10037,
10039,10061,10067,10069,10079,10091,10093,10099,
10103,10111,10133,10139,10141,10151,10159,10163,
10169,10177,10181,10193,10211,10223,10243,10247,
10253,10259,10267,10271,10273,10289,10301,10303,
10313,10321,10331,10333,10337,10343,10357,10369,
10391,10399,10427,10429,10433,10453,10457,10459,
10463,10477,10487,10499,10501,10513,10529,10531,
10559,10567,10589,10597,10601,10607,10613,10627,
10631,10639,10651,10657,10663,10667,10687,10691,
10709,10711,10723,10729,10733,10739,10753,10771,
10781,10789,10799,10831,10837,10847,10853,10859,
10861,10867,10883,10889,10891,10903,10909,10937,
10939,10949,10957,10973,10979,10987,10993,11003,
11027,11047,11057,11059,11069,11071,11083,11087,
11093,11113,11117,11119,11131,11149,11159,11161,
11171,11173,11177,11197,11213,11239,11243,11251,
11257,11261,11273,11279,11287,11299,11311,11317,
11321,11329,11351,11353,11369,11383,11393,11399,
11411,11423,11437,11443,11447,11467,11471,11483,
11489,11491,11497,11503,11519,11527,11549,11551,
11579,11587,11593,11597,11617,11621,11633,11657,
11677,11681,11689,11699,11701,11717,11719,11731,
11743,11777,11779,11783,11789,11801,11807,11813,
11821,11827,11831,11833,11839,11863,11867,11887,
11897,11903,11909,11923,11927,11933,11939,11941,
11953,11959,11969,11971,11981,11987,12007,12011,
12037,12041,12043,12049,12071,12073,12097,12101,
12107,12109,12113,12119,12143,12149,12157,12161,
12163,12197,12203,12211,12227,12239,12241,12251,
12253,12263,12269,12277,12281,12289,12301,12323,
12329,12343,12347,12373,12377,12379,12391,12401,
12409,12413,12421,12433,12437,12451,12457,12473,
12479,12487,12491,12497,12503,12511,12517,12527,
12539,12541,12547,12553,12569,12577,12583,12589,
12601,12611,12613,12619,12637,12641,12647,12653,
12659,12671,12689,12697,12703,12713,12721,12739,
12743,12757,12763,12781,12791,12799,12809,12821,
12823,12829,12841,12853,12889,12893,12899,12907,
12911,12917,12919,12923,12941,12953,12959,12967,
12973,12979,12983,13001,13003,13007,13009,13033,
13037,13043,13049,13063,13093,13099,13103,13109,
13121,13127,13147,13151,13159,13163,13171,13177,
13183,13187,13217,13219,13229,13241,13249,13259,
13267,13291,13297,13309,13313,13327,13331,13337,
13339,13367,13381,13397,13399,13411,13417,13421,
13441,13451,13457,13463,13469,13477,13487,13499,
13513,13523,13537,13553,13567,13577,13591,13597,
13613,13619,13627,13633,13649,13669,13679,13681,
13687,13691,13693,13697,13709,13711,13721,13723,
13729,13751,13757,13759,13763,13781,13789,13799,
13807,13829,13831,13841,13859,13873,13877,13879,
13883,13901,13903,13907,13913,13921,13931,13933,
13963,13967,13997,13999,14009,14011,14029,14033,
14051,14057,14071,14081,14083,14087,14107,14143,
14149,14153,14159,14173,14177,14197,14207,14221,
14243,14249,14251,14281,14293,14303,14321,14323,
14327,14341,14347,14369,14387,14389,14401,14407,
14411,14419,14423,14431,14437,14447,14449,14461,
14479,14489,14503,14519,14533,14537,14543,14549,
14551,14557,14561,14563,14591,14593,14621,14627,
14629,14633,14639,14653,14657,14669,14683,14699,
14713,14717,14723,14731,14737,14741,14747,14753,
14759,14767,14771,14779,14783,14797,14813,14821,
14827,14831,14843,14851,14867,14869,14879,14887,
14891,14897,14923,14929,14939,14947,14951,14957,
14969,14983,15013,15017,15031,15053,15061,15073,
15077,15083,15091,15101,15107,15121,15131,15137,
15139,15149,15161,15173,15187,15193,15199,15217,
15227,15233,15241,15259,15263,15269,15271,15277,
15287,15289,15299,15307,15313,15319,15329,15331,
15349,15359,15361,15373,15377,15383,15391,15401,
15413,15427,15439,15443,15451,15461,15467,15473,
15493,15497,15511,15527,15541,15551,15559,15569,
15581,15583,15601,15607,15619,15629,15641,15643,
15647,15649,15661,15667,15671,15679,15683,15727,
15731,15733,15737,15739,15749,15761,15767,15773,
15787,15791,15797,15803,15809,15817,15823,15859,
15877,15881,15887,15889,15901,15907,15913,15919,
15923,15937,15959,15971,15973,15991,16001,16007,
16033,16057,16061,16063,16067,16069,16073,16087,
16091,16097,16103,16111,16127,16139,16141,16183,
16187,16189,16193,16217,16223,16229,16231,16249,
16253,16267,16273,16301,16319,16333,16339,16349,
16361,16363,16369,16381,16411,16417,16421,16427,
16433,16447,16451,16453,16477,16481,16487,16493,
16519,16529,16547,16553,16561,16567,16573,16603,
16607,16619,16631,16633,16649,16651,16657,16661,
16673,16691,16693,16699,16703,16729,16741,16747,
16759,16763,16787,16811,16823,16829,16831,16843,
16871,16879,16883,16889,16901,16903,16921,16927,
16931,16937,16943,16963,16979,16981,16987,16993,
17011,17021,17027,17029,17033,17041,17047,17053,
17077,17093,17099,17107,17117,17123,17137,17159,
17167,17183,17189,17191,17203,17207,17209,17231,
17239,17257,17291,17293,17299,17317,17321,17327,
17333,17341,17351,17359,17377,17383,17387,17389,
17393,17401,17417,17419,17431,17443,17449,17467,
17471,17477,17483,17489,17491,17497,17509,17519,
17539,17551,17569,17573,17579,17581,17597,17599,
17609,17623,17627,17657,17659,17669,17681,17683,
17707,17713,17729,17737,17747,17749,17761,17783,
17789,17791,17807,17827,17837,17839,17851,17863,
17881,17891,17903,17909,17911,17921,17923,17929,
17939,17957,17959,17971,17977,17981,17987,17989,
18013,18041,18043,18047,18049,18059,18061,18077,
18089,18097,18119,18121,18127,18131,18133,18143,
18149,18169,18181,18191,18199,18211,18217,18223,
18229,18233,18251,18253,18257,18269,18287,18289,
18301,18307,18311,18313,18329,18341,18353,18367,
18371,18379,18397,18401,18413,18427,18433,18439,
18443,18451,18457,18461,18481,18493,18503,18517,
18521,18523,18539,18541,18553,18583,18587,18593,
18617,18637,18661,18671,18679,18691,18701,18713,
18719,18731,18743,18749,18757,18773,18787,18793,
18797,18803,18839,18859,18869,18899,18911,18913,
18917,18919,18947,18959,18973,18979,19001,19009,
19013,19031,19037,19051,19069,19073,19079,19081,
19087,19121,19139,19141,19157,19163,19181,19183,
19207,19211,19213,19219,19231,19237,19249,19259,
19267,19273,19289,19301,19309,19319,19333,19373,
19379,19381,19387,19391,19403,19417,19421,19423,
19427,19429,19433,19441,19447,19457,19463,19469,
19471,19477,19483,19489,19501,19507,19531,19541,
19543,19553,19559,19571,19577,19583,19597,19603,
19609,19661,19681,19687,19697,19699,19709,19717,
19727,19739,19751,19753,19759,19763,19777,19793,
19801,19813,19819,19841,19843,19853,19861,19867,
19889,19891,19913,19919,19927,19937,19949,19961,
19963,19973,19979,19991,19993,19997,20011,20021,
20023,20029,20047,20051,20063,20071,20089,20101,
20107,20113,20117,20123,20129,20143,20147,20149,
20161,20173,20177,20183,20201,20219,20231,20233,
20249,20261,20269,20287,20297,20323,20327,20333,
20341,20347,20353,20357,20359,20369,20389,20393,
20399,20407,20411,20431,20441,20443,20477,20479,
20483,20507,20509,20521,20533,20543,20549,20551,
20563,20593,20599,20611,20627,20639,20641,20663,
20681,20693,20707,20717,20719,20731,20743,20747,
20749,20753,20759,20771,20773,20789,20807,20809,
20849,20857,20873,20879,20887,20897,20899,20903,
20921,20929,20939,20947,20959,20963,20981,20983,
21001,21011,21013,21017,21019,21023,21031,21059,
21061,21067,21089,21101,21107,21121,21139,21143,
21149,21157,21163,21169,21179,21187,21191,21193,
21211,21221,21227,21247,21269,21277,21283,21313,
21317,21319,21323,21341,21347,21377,21379,21383,
21391,21397,21401,21407,21419,21433,21467,21481,
21487,21491,21493,21499,21503,21517,21521,21523,
21529,21557,21559,21563,21569,21577,21587,21589,
21599,21601,21611,21613,21617,21647,21649,21661,
21673,21683,21701,21713,21727,21737,21739,21751,
21757,21767,21773,21787,21799,21803,21817,21821,
21839,21841,21851,21859,21863,21871,21881,21893,
21911,21929,21937,21943,21961,21977,21991,21997,
22003,22013,22027,22031,22037,22039,22051,22063,
22067,22073,22079,22091,22093,22109,22111,22123,
22129,22133,22147,22153,22157,22159,22171,22189,
22193,22229,22247,22259,22271,22273,22277,22279,
22283,22291,22303,22307,22343,22349,22367,22369,
22381,22391,22397,22409,22433,22441,22447,22453,
22469,22481,22483,22501,22511,22531,22541,22543,
22549,22567,22571,22573,22613,22619,22621,22637,
22639,22643,22651,22669,22679,22691,22697,22699,
22709,22717,22721,22727,22739,22741,22751,22769,
22777,22783,22787,22807,22811,22817,22853,22859,
22861,22871,22877,22901,22907,22921,22937,22943,
22961,22963,22973,22993,23003,23011,23017,23021,
23027,23029,23039,23041,23053,23057,23059,23063,
23071,23081,23087,23099,23117,23131,23143,23159,
23167,23173,23189,23197,23201,23203,23209,23227,
23251,23269,23279,23291,23293,23297,23311,23321,
23327,23333,23339,23357,23369,23371,23399,23417,
23431,23447,23459,23473,23497,23509,23531,23537,
23539,23549,23557,23561,23563,23567,23581,23593,
23599,23603,23609,23623,23627,23629,23633,23663,
23669,23671,23677,23687,23689,23719,23741,23743,
23747,23753,23761,23767,23773,23789,23801,23813,
23819,23827,23831,23833,23857,23869,23873,23879,
23887,23893,23899,23909,23911,23917,23929,23957,
23971,23977,23981,23993,24001,24007,24019,24023,
24029,24043,24049,24061,24071,24077,24083,24091,
24097,24103,24107,24109,24113,24121,24133,24137,
24151,24169,24179,24181,24197,24203,24223,24229,
24239,24247,24251,24281,24317,24329,24337,24359,
24371,24373,24379,24391,24407,24413,24419,24421,
24439,24443,24469,24473,24481,24499,24509,24517,
24527,24533,24547,24551,24571,24593,24611,24623,
24631,24659,24671,24677,24683,24691,24697,24709,
24733,24749,24763,24767,24781,24793,24799,24809,
24821,24841,24847,24851,24859,24877,24889,24907,
24917,24919,24923,24943,24953,24967,24971,24977,
24979,24989,25013,25031,25033,25037,25057,25073,
25087,25097,25111,25117,25121,25127,25147,25153,
25163,25169,25171,25183,25189,25219,25229,25237,
25243,25247,25253,25261,25301,25303,25307,25309,
25321,25339,25343,25349,25357,25367,25373,25391,
25409,25411,25423,25439,25447,25453,25457,25463,
25469,25471,25523,25537,25541,25561,25577,25579,
25583,25589,25601,25603,25609,25621,25633,25639,
25643,25657,25667,25673,25679,25693,25703,25717,
25733,25741,25747,25759,25763,25771,25793,25799,
25801,25819,25841,25847,25849,25867,25873,25889,
25903,25913,25919,25931,25933,25939,25943,25951,
25969,25981,25997,25999,26003,26017,26021,26029,
26041,26053,26083,26099,26107,26111,26113,26119,
26141,26153,26161,26171,26177,26183,26189,26203,
26209,26227,26237,26249,26251,26261,26263,26267,
26293,26297,26309,26317,26321,26339,26347,26357,
26371,26387,26393,26399,26407,26417,26423,26431,
26437,26449,26459,26479,26489,26497,26501,26513,
26539,26557,26561,26573,26591,26597,26627,26633,
26641,26647,26669,26681,26683,26687,26693,26699,
26701,26711,26713,26717,26723,26729,26731,26737,
26759,26777,26783,26801,26813,26821,26833,26839,
26849,26861,26863,26879,26881,26891,26893,26903,
26921,26927,26947,26951,26953,26959,26981,26987,
26993,27011,27017,27031,27043,27059,27061,27067,
27073,27077,27091,27103,27107,27109,27127,27143,
27179,27191,27197,27211,27239,27241,27253,27259,
27271,27277,27281,27283,27299,27329,27337,27361,
27367,27397,27407,27409,27427,27431,27437,27449,
27457,27479,27481,27487,27509,27527,27529,27539,
27541,27551,27581,27583,27611,27617,27631,27647,
27653,27673,27689,27691,27697,27701,27733,27737,
27739,27743,27749,27751,27763,27767,27773,27779,
27791,27793,27799,27803,27809,27817,27823,27827,
27847,27851,27883,27893,27901,27917,27919,27941,
27943,27947,27953,27961,27967,27983,27997,28001,
28019,28027,28031,28051,28057,28069,28081,28087,
28097,28099,28109,28111,28123,28151,28163,28181,
28183,28201,28211,28219,28229,28277,28279,28283,
28289,28297,28307,28309,28319,28349,28351,28387,
28393,28403,28409,28411,28429,28433,28439,28447,
28463,28477,28493,28499,28513,28517,28537,28541,
28547,28549,28559,28571,28573,28579,28591,28597,
28603,28607,28619,28621,28627,28631,28643,28649,
28657,28661,28663,28669,28687,28697,28703,28711,
28723,28729,28751,28753,28759,28771,28789,28793,
28807,28813,28817,28837,28843,28859,28867,28871,
28879,28901,28909,28921,28927,28933,28949,28961,
28979,29009,29017,29021,29023,29027,29033,29059,
29063,29077,29101,29123,29129,29131,29137,29147,
29153,29167,29173,29179,29191,29201,29207,29209,
29221,29231,29243,29251,29269,29287,29297,29303,
29311,29327,29333,29339,29347,29363,29383,29387,
29389,29399,29401,29411,29423,29429,29437,29443,
29453,29473,29483,29501,29527,29531,29537,29567,
29569,29573,29581,29587,29599,29611,29629,29633,
29641,29663,29669,29671,29683,29717,29723,29741,
29753,29759,29761,29789,29803,29819,29833,29837,
29851,29863,29867,29873,29879,29881,29917,29921,
29927,29947,29959,29983,29989,30011,30013,30029,
30047,30059,30071,30089,30091,30097,30103,30109,
30113,30119,30133,30137,30139,30161,30169,30181,
30187,30197,30203,30211,30223,30241,30253,30259,
30269,30271,30293,30307,30313,30319,30323,30341,
30347,30367,30389,30391,30403,30427,30431,30449,
30467,30469,30491,30493,30497,30509,30517,30529,
30539,30553,30557,30559,30577,30593,30631,30637,
30643,30649,30661,30671,30677,30689,30697,30703,
30707,30713,30727,30757,30763,30773,30781,30803,
30809,30817,30829,30839,30841,30851,30853,30859,
30869,30871,30881,30893,30911,30931,30937,30941,
30949,30971,30977,30983,31013,31019,31033,31039,
31051,31063,31069,31079,31081,31091,31121,31123,
31139,31147,31151,31153,31159,31177,31181,31183,
31189,31193,31219,31223,31231,31237,31247,31249,
31253,31259,31267,31271,31277,31307,31319,31321,
31327,31333,31337,31357,31379,31387,31391,31393,
31397,31469,31477,31481,31489,31511,31513,31517,
31531,31541,31543,31547,31567,31573,31583,31601,
31607,31627,31643,31649,31657,31663,31667,31687,
31699,31721,31723,31727,31729,31741,31751,31769,
31771,31793,31799,31817,31847,31849,31859,31873,
31883,31891,31907,31957,31963,31973,31981,31991,
32003,32009,32027,32029,32051,32057,32059,32063,
32069,32077,32083,32089,32099,32117,32119,32141,
32143,32159,32173,32183,32189,32191,32203,32213,
32233,32237,32251,32257,32261,32297,32299,32303,
32309,32321,32323,32327,32341,32353,32359,32363,
32369,32371,32377,32381,32401,32411,32413,32423,
32429,32441,32443,32467,32479,32491,32497,32503,
32507,32531,32533,32537,32561,32563,32569,32573,
32579,32587,32603,32609,32611,32621,32633,32647,
32653,32687,32693,32707,32713,32717,32719,32749,
32771,32779,32783,32789,32797,32801,32803,32831,
32833,32839,32843,32869,32887,32909,32911,32917,
32933,32939,32941,32957,32969,32971,32983,32987,
32993,32999,33013,33023,33029,33037,33049,33053,
33071,33073,33083,33091,33107,33113,33119,33149,
33151,33161,33179,33181,33191,33199,33203,33211,
33223,33247,33287,33289,33301,33311,33317,33329,
33331,33343,33347,33349,33353,33359,33377,33391,
33403,33409,33413,33427,33457,33461,33469,33479,
33487,33493,33503,33521,33529,33533,33547,33563,
33569,33577,33581,33587,33589,33599,33601,33613,
33617,33619,33623,33629,33637,33641,33647,33679,
33703,33713,33721,33739,33749,33751,33757,33767,
33769,33773,33791,33797,33809,33811,33827,33829,
33851,33857,33863,33871,33889,33893,33911,33923,
33931,33937,33941,33961,33967,33997,34019,34031,
34033,34039,34057,34061,34123,34127,34129,34141,
34147,34157,34159,34171,34183,34211,34213,34217,
34231,34253,34259,34261,34267,34273,34283,34297,
34301,34303,34313,34319,34327,34337,34351,34361,
34367,34369,34381,34403,34421,34429,34439,34457,
34469,34471,34483,34487,34499,34501,34511,34513,
34519,34537,34543,34549,34583,34589,34591,34603,
34607,34613,34631,34649,34651,34667,34673,34679,
34687,34693,34703,34721,34729,34739,34747,34757,
34759,34763,34781,34807,34819,34841,34843,34847,
34849,34871,34877,34883,34897,34913,34919,34939,
34949,34961,34963,34981,35023,35027,35051,35053,
35059,35069,35081,35083,35089,35099,35107,35111,
35117,35129,35141,35149,35153,35159,35171,35201,
35221,35227,35251,35257,35267,35279,35281,35291,
35311,35317,35323,35327,35339,35353,35363,35381,
35393,35401,35407,35419,35423,35437,35447,35449,
35461,35491,35507,35509,35521,35527,35531,35533,
35537,35543,35569,35573,35591,35593,35597,35603,
35617,35671,35677,35729,35731,35747,35753,35759,
35771,35797,35801,35803,35809,35831,35837,35839,
35851,35863,35869,35879,35897,35899,35911,35923,
35933,35951,35963,35969,35977,35983,35993,35999,
36007,36011,36013,36017,36037,36061,36067,36073,
36083,36097,36107,36109,36131,36137,36151,36161,
36187,36191,36209,36217,36229,36241,36251,36263,
36269,36277,36293,36299,36307,36313,36319,36341,
36343,36353,36373,36383,36389,36433,36451,36457,
36467,36469,36473,36479,36493,36497,36523,36527,
36529,36541,36551,36559,36563,36571,36583,36587,
36599,36607,36629,36637,36643,36653,36671,36677,
36683,36691,36697,36709,36713,36721,36739,36749,
36761,36767,36779,36781,36787,36791,36793,36809,
36821,36833,36847,36857,36871,36877,36887,36899,
36901,36913,36919,36923,36929,36931,36943,36947,
36973,36979,36997,37003,37013,37019,37021,37039,
37049,37057,37061,37087,37097,37117,37123,37139,
37159,37171,37181,37189,37199,37201,37217,37223,
37243,37253,37273,37277,37307,37309,37313,37321,
37337,37339,37357,37361,37363,37369,37379,37397,
37409,37423,37441,37447,37463,37483,37489,37493,
37501,37507,37511,37517,37529,37537,37547,37549,
37561,37567,37571,37573,37579,37589,37591,37607,
37619,37633,37643,37649,37657,37663,37691,37693,
37699,37717,37747,37781,37783,37799,37811,37813,
37831,37847,37853,37861,37871,37879,37889,37897,
37907,37951,37957,37963,37967,37987,37991,37993,
37997,38011,38039,38047,38053,38069,38083,38113,
38119,38149,38153,38167,38177,38183,38189,38197,
38201,38219,38231,38237,38239,38261,38273,38281,
38287,38299,38303,38317,38321,38327,38329,38333,
38351,38371,38377,38393,38431,38447,38449,38453,
38459,38461,38501,38543,38557,38561,38567,38569,
38593,38603,38609,38611,38629,38639,38651,38653,
38669,38671,38677,38693,38699,38707,38711,38713,
38723,38729,38737,38747,38749,38767,38783,38791,
38803,38821,38833,38839,38851,38861,38867,38873,
38891,38903,38917,38921,38923,38933,38953,38959,
38971,38977,38993,39019,39023,39041,39043,39047,
39079,39089,39097,39103,39107,39113,39119,39133,
39139,39157,39161,39163,39181,39191,39199,39209,
39217,39227,39229,39233,39239,39241,39251,39293,
39301,39313,39317,39323,39341,39343,39359,39367,
39371,39373,39383,39397,39409,39419,39439,39443,
39451,39461,39499,39503,39509,39511,39521,39541,
39551,39563,39569,39581,39607,39619,39623,39631,
39659,39667,39671,39679,39703,39709,39719,39727,
39733,39749,39761,39769,39779,39791,39799,39821,
39827,39829,39839,39841,39847,39857,39863,39869,
39877,39883,39887,39901,39929,39937,39953,39971,
39979,39983,39989,40009,40013,40031,40037,40039,
40063,40087,40093,40099,40111,40123,40127,40129,
40151,40153,40163,40169,40177,40189,40193,40213,
40231,40237,40241,40253,40277,40283,40289,40343,
40351,40357,40361,40387,40423,40427,40429,40433,
40459,40471,40483,40487,40493,40499,40507,40519,
40529,40531,40543,40559,40577,40583,40591,40597,
40609,40627,40637,40639,40693,40697,40699,40709,
40739,40751,40759,40763,40771,40787,40801,40813,
40819,40823,40829,40841,40847,40849,40853,40867,
40879,40883,40897,40903,40927,40933,40939,40949,
40961,40973,40993,41011,41017,41023,41039,41047,
41051,41057,41077,41081,41113,41117,41131,41141,
41143,41149,41161,41177,41179,41183,41189,41201,
41203,41213,41221,41227,41231,41233,41243,41257,
41263,41269,41281,41299,41333,41341,41351,41357,
41381,41387,41389,41399,41411,41413,41443,41453,
41467,41479,41491,41507,41513,41519,41521,41539,
41543,41549,41579,41593,41597,41603,41609,41611,
41617,41621,41627,41641,41647,41651,41659,41669,
41681,41687,41719,41729,41737,41759,41761,41771,
41777,41801,41809,41813,41843,41849,41851,41863,
41879,41887,41893,41897,41903,41911,41927,41941,
41947,41953,41957,41959,41969,41981,41983,41999,
42013,42017,42019,42023,42043,42061,42071,42073,
42083,42089,42101,42131,42139,42157,42169,42179,
42181,42187,42193,42197,42209,42221,42223,42227,
42239,42257,42281,42283,42293,42299,42307,42323,
42331,42337,42349,42359,42373,42379,42391,42397,
42403,42407,42409,42433,42437,42443,42451,42457,
42461,42463,42467,42473,42487,42491,42499,42509,
42533,42557,42569,42571,42577,42589,42611,42641,
42643,42649,42667,42677,42683,42689,42697,42701,
42703,42709,42719,42727,42737,42743,42751,42767,
42773,42787,42793,42797,42821,42829,42839,42841,
42853,42859,42863,42899,42901,42923,42929,42937,
42943,42953,42961,42967,42979,42989,43003,43013,
43019,43037,43049,43051,43063,43067,43093,43103,
43117,43133,43151,43159,43177,43189,43201,43207,
43223,43237,43261,43271,43283,43291,43313,43319,
43321,43331,43391,43397,43399,43403,43411,43427,
43441,43451,43457,43481,43487,43499,43517,43541,
43543,43573,43577,43579,43591,43597,43607,43609,
43613,43627,43633,43649,43651,43661,43669,43691,
43711,43717,43721,43753,43759,43777,43781,43783,
43787,43789,43793,43801,43853,43867,43889,43891,
43913,43933,43943,43951,43961,43963,43969,43973,
43987,43991,43997,44017,44021,44027,44029,44041,
44053,44059,44071,44087,44089,44101,44111,44119,
44123,44129,44131,44159,44171,44179,44189,44201,
44203,44207,44221,44249,44257,44263,44267,44269,
44273,44279,44281,44293,44351,44357,44371,44381,
44383,44389,44417,44449,44453,44483,44491,44497,
44501,44507,44519,44531,44533,44537,44543,44549,
44563,44579,44587,44617,44621,44623,44633,44641,
44647,44651,44657,44683,44687,44699,44701,44711,
44729,44741,44753,44771,44773,44777,44789,44797,
44809,44819,44839,44843,44851,44867,44879,44887,
44893,44909,44917,44927,44939,44953,44959,44963,
44971,44983,44987,45007,45013,45053,45061,45077,
45083,45119,45121,45127,45131,45137,45139,45161,
45179,45181,45191,45197,45233,45247,45259,45263,
45281,45289,45293,45307,45317,45319,45329,45337,
45341,45343,45361,45377,45389,45403,45413,45427,
45433,45439,45481,45491,45497,45503,45523,45533,
45541,45553,45557,45569,45587,45589,45599,45613,
45631,45641,45659,45667,45673,45677,45691,45697,
45707,45737,45751,45757,45763,45767,45779,45817,
45821,45823,45827,45833,45841,45853,45863,45869,
45887,45893,45943,45949,45953,45959,45971,45979,
45989,46021,46027,46049,46051,46061,46073,46091,
46093,46099,46103,46133,46141,46147,46153,46171,
46181,46183,46187,46199,46219,46229,46237,46261,
46271,46273,46279,46301,46307,46309,46327,46337,
46349,46351,46381,46399,46411,46439,46441,46447,
46451,46457,46471,46477,46489,46499,46507,46511,
46523,46549,46559,46567,46573,46589,46591,46601,
46619,46633,46639,46643,46649,46663,46679,46681,
46687,46691,46703,46723,46727,46747,46751,46757,
46769,46771,46807,46811,46817,46819,46829,46831,
46853,46861,46867,46877,46889,46901,46919,46933,
46957,46993,46997,47017,47041,47051,47057,47059,
47087,47093,47111,47119,47123,47129,47137,47143,
47147,47149,47161,47189,47207,47221,47237,47251,
47269,47279,47287,47293,47297,47303,47309,47317,
47339,47351,47353,47363,47381,47387,47389,47407,
47417,47419,47431,47441,47459,47491,47497,47501,
47507,47513,47521,47527,47533,47543,47563,47569,
47581,47591,47599,47609,47623,47629,47639,47653,
47657,47659,47681,47699,47701,47711,47713,47717,
47737,47741,47743,47777,47779,47791,47797,47807,
47809,47819,47837,47843,47857,47869,47881,47903,
47911,47917,47933,47939,47947,47951,47963,47969,
47977,47981,48017,48023,48029,48049,48073,48079,
48091,48109,48119,48121,48131,48157,48163,48179,
48187,48193,48197,48221,48239,48247,48259,48271,
48281,48299,48311,48313,48337,48341,48353,48371,
48383,48397,48407,48409,48413,48437,48449,48463,
48473,48479,48481,48487,48491,48497,48523,48527,
48533,48539,48541,48563,48571,48589,48593,48611,
48619,48623,48647,48649,48661,48673,48677,48679,
48731,48733,48751,48757,48761,48767,48779,48781,
48787,48799,48809,48817,48821,48823,48847,48857,
48859,48869,48871,48883,48889,48907,48947,48953,
48973,48989,48991,49003,49009,49019,49031,49033,
49037,49043,49057,49069,49081,49103,49109,49117,
49121,49123,49139,49157,49169,49171,49177,49193,
49199,49201,49207,49211,49223,49253,49261,49277,
49279,49297,49307,49331,49333,49339,49363,49367,
49369,49391,49393,49409,49411,49417,49429,49433,
49451,49459,49463,49477,49481,49499,49523,49529,
49531,49537,49547,49549,49559,49597,49603,49613,
49627,49633,49639,49663,49667,49669,49681,49697,
49711,49727,49739,49741,49747,49757,49783,49787,
49789,49801,49807,49811,49823,49831,49843,49853,
49871,49877,49891,49919,49921,49927,49937,49939,
49943,49957,49991,49993,49999,50021,50023,50033,
50047,50051,50053,50069,50077,50087,50093,50101,
50111,50119,50123,50129,50131,50147,50153,50159,
50177,50207,50221,50227,50231,50261,50263,50273,
50287,50291,50311,50321,50329,50333,50341,50359,
50363,50377,50383,50387,50411,50417,50423,50441,
50459,50461,50497,50503,50513,50527,50539,50543,
50549,50551,50581,50587,50591,50593,50599,50627,
50647,50651,50671,50683,50707,50723,50741,50753,
50767,50773,50777,50789,50821,50833,50839,50849,
50857,50867,50873,50891,50893,50909,50923,50929,
50951,50957,50969,50971,50989,50993,51001,51031,
51043,51047,51059,51061,51071,51109,51131,51133,
51137,51151,51157,51169,51193,51197,51199,51203,
51217,51229,51239,51241,51257,51263,51283,51287,
51307,51329,51341,51343,51347,51349,51361,51383,
51407,51413,51419,51421,51427,51431,51437,51439,
51449,51461,51473,51479,51481,51487,51503,51511,
51517,51521,51539,51551,51563,51577,51581,51593,
51599,51607,51613,51631,51637,51647,51659,51673,
51679,51683,51691,51713,51719,51721,51749,51767,
51769,51787,51797,51803,51817,51827,51829,51839,
51853,51859,51869,51871,51893,51899,51907,51913,
51929,51941,51949,51971,51973,51977,51991,52009,
52021,52027,52051,52057,52067,52069,52081,52103,
52121,52127,52147,52153,52163,52177,52181,52183,
52189,52201,52223,52237,52249,52253,52259,52267,
52289,52291,52301,52313,52321,52361,52363,52369,
52379,52387,52391,52433,52453,52457,52489,52501,
52511,52517,52529,52541,52543,52553,52561,52567,
52571,52579,52583,52609,52627,52631,52639,52667,
52673,52691,52697,52709,52711,52721,52727,52733,
52747,52757,52769,52783,52807,52813,52817,52837,
52859,52861,52879,52883,52889,52901,52903,52919,
52937,52951,52957,52963,52967,52973,52981,52999,
53003,53017,53047,53051,53069,53077,53087,53089,
53093,53101,53113,53117,53129,53147,53149,53161,
53171,53173,53189,53197,53201,53231,53233,53239,
53267,53269,53279,53281,53299,53309,53323,53327,
53353,53359,53377,53381,53401,53407,53411,53419,
53437,53441,53453,53479,53503,53507,53527,53549,
53551,53569,53591,53593,53597,53609,53611,53617,
53623,53629,53633,53639,53653,53657,53681,53693,
53699,53717,53719,53731,53759,53773,53777,53783,
53791,53813,53819,53831,53849,53857,53861,53881,
53887,53891,53897,53899,53917,53923,53927,53939,
53951,53959,53987,53993,54001,54011,54013,54037,
54049,54059,54083,54091,54101,54121,54133,54139,
54151,54163,54167,54181,54193,54217,54251,54269,
54277,54287,54293,54311,54319,54323,54331,54347,
54361,54367,54371,54377,54401,54403,54409,54413,
54419,54421,54437,54443,54449,54469,54493,54497,
54499,54503,54517,54521,54539,54541,54547,54559,
54563,54577,54581,54583,54601,54617,54623,54629,
54631,54647,54667,54673,54679,54709,54713,54721,
54727,54751,54767,54773,54779,54787,54799,54829,
54833,54851,54869,54877,54881,54907,54917,54919,
54941,54949,54959,54973,54979,54983,55001,55009,
55021,55049,55051,55057,55061,55073,55079,55103,
55109,55117,55127,55147,55163,55171,55201,55207,
55213,55217,55219,55229,55243,55249,55259,55291,
55313,55331,55333,55337,55339,55343,55351,55373,
55381,55399,55411,55439,55441,55457,55469,55487,
55501,55511,55529,55541,55547,55579,55589,55603,
55609,55619,55621,55631,55633,55639,55661,55663,
55667,55673,55681,55691,55697,55711,55717,55721,
55733,55763,55787,55793,55799,55807,55813,55817,
55819,55823,55829,55837,55843,55849,55871,55889,
55897,55901,55903,55921,55927,55931,55933,55949,
55967,55987,55997,56003,56009,56039,56041,56053,
56081,56087,56093,56099,56101,56113,56123,56131,
56149,56167,56171,56179,56197,56207,56209,56237,
56239,56249,56263,56267,56269,56299,56311,56333,
56359,56369,56377,56383,56393,56401,56417,56431,
56437,56443,56453,56467,56473,56477,56479,56489,
56501,56503,56509,56519,56527,56531,56533,56543,
56569,56591,56597,56599,56611,56629,56633,56659,
56663,56671,56681,56687,56701,56711,56713,56731,
56737,56747,56767,56773,56779,56783,56807,56809,
56813,56821,56827,56843,56857,56873,56891,56893,
56897,56909,56911,56921,56923,56929,56941,56951,
56957,56963,56983,56989,56993,56999,57037,57041,
57047,57059,57073,57077,57089,57097,57107,57119,
57131,57139,57143,57149,57163,57173,57179,57191,
57193,57203,57221,57223,57241,57251,57259,57269,
57271,57283,57287,57301,57329,57331,57347,57349,
57367,57373,57383,57389,57397,57413,57427,57457,
57467,57487,57493,57503,57527,57529,57557,57559,
57571,57587,57593,57601,57637,57641,57649,57653,
57667,57679,57689,57697,57709,57713,57719,57727,
57731,57737,57751,57773,57781,57787,57791,57793,
57803,57809,57829,57839,57847,57853,57859,57881,
57899,57901,57917,57923,57943,57947,57973,57977,
57991,58013,58027,58031,58043,58049,58057,58061,
58067,58073,58099,58109,58111,58129,58147,58151,
58153,58169,58171,58189,58193,58199,58207,58211,
58217,58229,58231,58237,58243,58271,58309,58313,
58321,58337,58363,58367,58369,58379,58391,58393,
58403,58411,58417,58427,58439,58441,58451,58453,
58477,58481,58511,58537,58543,58549,58567,58573,
58579,58601,58603,58613,58631,58657,58661,58679,
58687,58693,58699,58711,58727,58733,58741,58757,
58763,58771,58787,58789,58831,58889,58897,58901,
58907,58909,58913,58921,58937,58943,58963,58967,
58979,58991,58997,59009,59011,59021,59023,59029,
59051,59053,59063,59069,59077,59083,59093,59107,
59113,59119,59123,59141,59149,59159,59167,59183,
59197,59207,59209,59219,59221,59233,59239,59243,
59263,59273,59281,59333,59341,59351,59357,59359,
59369,59377,59387,59393,59399,59407,59417,59419,
59441,59443,59447,59453,59467,59471,59473,59497,
59509,59513,59539,59557,59561,59567,59581,59611,
59617,59621,59627,59629,59651,59659,59663,59669,
59671,59693,59699,59707,59723,59729,59743,59747,
59753,59771,59779,59791,59797,59809,59833,59863,
59879,59887,59921,59929,59951,59957,59971,59981,
59999,60013,60017,60029,60037,60041,60077,60083,
60089,60091,60101,60103,60107,60127,60133,60139,
60149,60161,60167,60169,60209,60217,60223,60251,
60257,60259,60271,60289,60293,60317,60331,60337,
60343,60353,60373,60383,60397,60413,60427,60443,
60449,60457,60493,60497,60509,60521,60527,60539,
60589,60601,60607,60611,60617,60623,60631,60637,
60647,60649,60659,60661,60679,60689,60703,60719,
60727,60733,60737,60757,60761,60763,60773,60779,
60793,60811,60821,60859,60869,60887,60889,60899,
60901,60913,60917,60919,60923,60937,60943,60953,
60961,61001,61007,61027,61031,61043,61051,61057,
61091,61099,61121,61129,61141,61151,61153,61169,
61211,61223,61231,61253,61261,61283,61291,61297,
61331,61333,61339,61343,61357,61363,61379,61381,
61403,61409,61417,61441,61463,61469,61471,61483,
61487,61493,61507,61511,61519,61543,61547,61553,
61559,61561,61583,61603,61609,61613,61627,61631,
61637,61643,61651,61657,61667,61673,61681,61687,
61703,61717,61723,61729,61751,61757,61781,61813,
61819,61837,61843,61861,61871,61879,61909,61927,
61933,61949,61961,61967,61979,61981,61987,61991,
62003,62011,62017,62039,62047,62053,62057,62071,
62081,62099,62119,62129,62131,62137,62141,62143,
62171,62189,62191,62201,62207,62213,62219,62233,
62273,62297,62299,62303,62311,62323,62327,62347,
62351,62383,62401,62417,62423,62459,62467,62473,
62477,62483,62497,62501,62507,62533,62539,62549,
62563,62581,62591,62597,62603,62617,62627,62633,
62639,62653,62659,62683,62687,62701,62723,62731,
62743,62753,62761,62773,62791,62801,62819,62827,
62851,62861,62869,62873,62897,62903,62921,62927,
62929,62939,62969,62971,62981,62983,62987,62989,
63029,63031,63059,63067,63073,63079,63097,63103,
63113,63127,63131,63149,63179,63197,63199,63211,
63241,63247,63277,63281,63299,63311,63313,63317,
63331,63337,63347,63353,63361,63367,63377,63389,
63391,63397,63409,63419,63421,63439,63443,63463,
63467,63473,63487,63493,63499,63521,63527,63533,
63541,63559,63577,63587,63589,63599,63601,63607,
63611,63617,63629,63647,63649,63659,63667,63671,
63689,63691,63697,63703,63709,63719,63727,63737,
63743,63761,63773,63781,63793,63799,63803,63809,
63823,63839,63841,63853,63857,63863,63901,63907,
63913,63929,63949,63977,63997,64007,64013,64019,
64033,64037,64063,64067,64081,64091,64109,64123,
64151,64153,64157,64171,64187,64189,64217,64223,
64231,64237,64271,64279,64283,64301,64303,64319,
64327,64333,64373,64381,64399,64403,64433,64439,
64451,64453,64483,64489,64499,64513,64553,64567,
64577,64579,64591,64601,64609,64613,64621,64627,
64633,64661,64663,64667,64679,64693,64709,64717,
64747,64763,64781,64783,64793,64811,64817,64849,
64853,64871,64877,64879,64891,64901,64919,64921,
64927,64937,64951,64969,64997,65003,65011,65027,
65029,65033,65053,65063,65071,65089,65099,65101,
65111,65119,65123,65129,65141,65147,65167,65171,
65173,65179,65183,65203,65213,65239,65257,65267,
65269,65287,65293,65309,65323,65327,65353,65357,
65371,65381,65393,65407,65413,65419,65423,65437,
65447,65449,65479,65497,65519,65521,65537,65539,
65543,65551,65557,65563,65579,65581,65587,65599,
65609,65617,65629,65633,65647,65651,65657,65677,
65687,65699,65701,65707,65713,65717,65719,65729,
65731,65761,65777,65789,65809,65827,65831,65837,
65839,65843,65851,65867,65881,65899,65921,65927,
65929,65951,65957,65963,65981,65983,65993,66029,
66037,66041,66047,66067,66071,66083,66089,66103,
66107,66109,66137,66161,66169,66173,66179,66191,
66221,66239,66271,66293,66301,66337,66343,66347,
66359,66361,66373,66377,66383,66403,66413,66431,
66449,66457,66463,66467,66491,66499,66509,66523,
66529,66533,66541,66553,66569,66571,66587,66593,
66601,66617,66629,66643,66653,66683,66697,66701,
66713,66721,66733,66739,66749,66751,66763,66791,
66797,66809,66821,66841,66851,66853,66863,66877,
66883,66889,66919,66923,66931,66943,66947,66949,
66959,66973,66977,67003,67021,67033,67043,67049,
67057,67061,67073,67079,67103,67121,67129,67139,
67141,67153,67157,67169,67181,67187,67189,67211,
67213,67217,67219,67231,67247,67261,67271,67273,
67289,67307,67339,67343,67349,67369,67391,67399,
67409,67411,67421,67427,67429,67433,67447,67453,
67477,67481,67489,67493,67499,67511,67523,67531,
67537,67547,67559,67567,67577,67579,67589,67601,
67607,67619,67631,67651,67679,67699,67709,67723,
67733,67741,67751,67757,67759,67763,67777,67783,
67789,67801,67807,67819,67829,67843,67853,67867,
67883,67891,67901,67927,67931,67933,67939,67943,
67957,67961,67967,67979,67987,67993,68023,68041,
68053,68059,68071,68087,68099,68111,68113,68141,
68147,68161,68171,68207,68209,68213,68219,68227,
68239,68261,68279,68281,68311,68329,68351,68371,
68389,68399,68437,68443,68447,68449,68473,68477,
68483,68489,68491,68501,68507,68521,68531,68539,
68543,68567,68581,68597,68611,68633,68639,68659,
68669,68683,68687,68699,68711,68713,68729,68737,
68743,68749,68767,68771,68777,68791,68813,68819,
68821,68863,68879,68881,68891,68897,68899,68903,
68909,68917,68927,68947,68963,68993,69001,69011,
69019,69029,69031,69061,69067,69073,69109,69119,
69127,69143,69149,69151,69163,69191,69193,69197,
69203,69221,69233,69239,69247,69257,69259,69263,
69313,69317,69337,69341,69371,69379,69383,69389,
69401,69403,69427,69431,69439,69457,69463,69467,
69473,69481,69491,69493,69497,69499,69539,69557,
69593,69623,69653,69661,69677,69691,69697,69709,
69737,69739,69761,69763,69767,69779,69809,69821,
69827,69829,69833,69847,69857,69859,69877,69899,
69911,69929,69931,69941,69959,69991,69997,70001,
70003,70009,70019,70039,70051,70061,70067,70079,
70099,70111,70117,70121,70123,70139,70141,70157,
70163,70177,70181,70183,70199,70201,70207,70223,
70229,70237,70241,70249,70271,70289,70297,70309,
70313,70321,70327,70351,70373,70379,70381,70393,
70423,70429,70439,70451,70457,70459,70481,70487,
70489,70501,70507,70529,70537,70549,70571,70573,
70583,70589,70607,70619,70621,70627,70639,70657,
70663,70667,70687,70709,70717,70729,70753,70769,
70783,70793,70823,70841,70843,70849,70853,70867,
70877,70879,70891,70901,70913,70919,70921,70937,
70949,70951,70957,70969,70979,70981,70991,70997,
70999,71011,71023,71039,71059,71069,71081,71089,
71119,71129,71143,71147,71153,71161,71167,71171,
71191,71209,71233,71237,71249,71257,71261,71263,
71287,71293,71317,71327,71329,71333,71339,71341,
71347,71353,71359,71363,71387,71389,71399,71411,
71413,71419,71429,71437,71443,71453,71471,71473,
71479,71483,71503,71527,71537,71549,71551,71563,
71569,71593,71597,71633,71647,71663,71671,71693,
71699,71707,71711,71713,71719,71741,71761,71777,
71789,71807,71809,71821,71837,71843,71849,71861,
71867,71879,71881,71887,71899,71909,71917,71933,
71941,71947,71963,71971,71983,71987,71993,71999,
72019,72031,72043,72047,72053,72073,72077,72089,
72091,72101,72103,72109,72139,72161,72167,72169,
72173,72211,72221,72223,72227,72229,72251,72253,
72269,72271,72277,72287,72307,72313,72337,72341,
72353,72367,72379,72383,72421,72431,72461,72467,
72469,72481,72493,72497,72503,72533,72547,72551,
72559,72577,72613,72617,72623,72643,72647,72649,
72661,72671,72673,72679,72689,72701,72707,72719,
72727,72733,72739,72763,72767,72797,72817,72823,
72859,72869,72871,72883,72889,72893,72901,72907,
72911,72923,72931,72937,72949,72953,72959,72973,
72977,72997,73009,73013,73019,73037,73039,73043,
73061,73063,73079,73091,73121,73127,73133,73141,
73181,73189,73237,73243,73259,73277,73291,73303,
73309,73327,73331,73351,73361,73363,73369,73379,
73387,73417,73421,73433,73453,73459,73471,73477,
73483,73517,73523,73529,73547,73553,73561,73571,
73583,73589,73597,73607,73609,73613,73637,73643,
73651,73673,73679,73681,73693,73699,73709,73721,
73727,73751,73757,73771,73783,73819,73823,73847,
73849,73859,73867,73877,73883,73897,73907,73939,
73943,73951,73961,73973,73999,74017,74021,74027,
74047,74051,74071,74077,74093,74099,74101,74131,
74143,74149,74159,74161,74167,74177,74189,74197,
74201,74203,74209,74219,74231,74257,74279,74287,
74293,74297,74311,74317,74323,74353,74357,74363,
74377,74381,74383,74411,74413,74419,74441,74449,
74453,74471,74489,74507,74509,74521,74527,74531,
74551,74561,74567,74573,74587,74597,74609,74611,
74623,74653,74687,74699,74707,74713,74717,74719,
74729,74731,74747,74759,74761,74771,74779,74797,
74821,74827,74831,74843,74857,74861,74869,74873,
74887,74891,74897,74903,74923,74929,74933,74941,
74959,75011,75013,75017,75029,75037,75041,75079,
75083,75109,75133,75149,75161,75167,75169,75181,
75193,75209,75211,75217,75223,75227,75239,75253,
75269,75277,75289,75307,75323,75329,75337,75347,
75353,75367,75377,75389,75391,75401,75403,75407,
75431,75437,75479,75503,75511,75521,75527,75533,
75539,75541,75553,75557,75571,75577,75583,75611,
75617,75619,75629,75641,75653,75659,75679,75683,
75689,75703,75707,75709,75721,75731,75743,75767,
75773,75781,75787,75793,75797,75821,75833,75853,
75869,75883,75913,75931,75937,75941,75967,75979,
75983,75989,75991,75997,76001,76003,76031,76039,
76079,76081,76091,76099,76103,76123,76129,76147,
76157,76159,76163,76207,76213,76231,76243,76249,
76253,76259,76261,76283,76289,76303,76333,76343,
76367,76369,76379,76387,76403,76421,76423,76441,
76463,76471,76481,76487,76493,76507,76511,76519,
76537,76541,76543,76561,76579,76597,76603,76607,
76631,76649,76651,76667,76673,76679,76697,76717,
76733,76753,76757,76771,76777,76781,76801,76819,
76829,76831,76837,76847,76871,76873,76883,76907,
76913,76919,76943,76949,76961,76963,76991,77003,
77017,77023,77029,77041,77047,77069,77081,77093,
77101,77137,77141,77153,77167,77171,77191,77201,
77213,77237,77239,77243,77249,77261,77263,77267,
77269,77279,77291,77317,77323,77339,77347,77351,
77359,77369,77377,77383,77417,77419,77431,77447,
77471,77477,77479,77489,77491,77509,77513,77521,
77527,77543,77549,77551,77557,77563,77569,77573,
77587,77591,77611,77617,77621,77641,77647,77659,
77681,77687,77689,77699,77711,77713,77719,77723,
77731,77743,77747,77761,77773,77783,77797,77801,
77813,77839,77849,77863,77867,77893,77899,77929,
77933,77951,77969,77977,77983,77999,78007,78017,
78031,78041,78049,78059,78079,78101,78121,78137,
78139,78157,78163,78167,78173,78179,78191,78193,
78203,78229,78233,78241,78259,78277,78283,78301,
78307,78311,78317,78341,78347,78367,78401,78427,
78437,78439,78467,78479,78487,78497,78509,78511,
78517,78539,78541,78553,78569,78571,78577,78583,
78593,78607,78623,78643,78649,78653,78691,78697,
78707,78713,78721,78737,78779,78781,78787,78791,
78797,78803,78809,78823,78839,78853,78857,78877,
78887,78889,78893,78901,78919,78929,78941,78977,
78979,78989,79031,79039,79043,79063,79087,79103,
79111,79133,79139,79147,79151,79153,79159,79181,
79187,79193,79201,79229,79231,79241,79259,79273,
79279,79283,79301,79309,79319,79333,79337,79349,
79357,79367,79379,79393,79397,79399,79411,79423,
79427,79433,79451,79481,79493,79531,79537,79549,
79559,79561,79579,79589,79601,79609,79613,79621,
79627,79631,79633,79657,79669,79687,79691,79693,
79697,79699,79757,79769,79777,79801,79811,79813,
79817,79823,79829,79841,79843,79847,79861,79867,
79873,79889,79901,79903,79907,79939,79943,79967,
79973,79979,79987,79997,79999,80021,80039,80051,
80071,80077,80107,80111,80141,80147,80149,80153,
80167,80173,80177,80191,80207,80209,80221,80231,
80233,80239,80251,80263,80273,80279,80287,80309,
80317,80329,80341,80347,80363,80369,80387,80407,
80429,80447,80449,80471,80473,80489,80491,80513,
80527,80537,80557,80567,80599,80603,80611,80621,
80627,80629,80651,80657,80669,80671,80677,80681,
80683,80687,80701,80713,80737,80747,80749,80761,
80777,80779,80783,80789,80803,80809,80819,80831,
80833,80849,80863,80897,80909,80911,80917,80923,
80929,80933,80953,80963,80989,81001,81013,81017,
81019,81023,81031,81041,81043,81047,81049,81071,
81077,81083,81097,81101,81119,81131,81157,81163,
81173,81181,81197,81199,81203,81223,81233,81239,
81281,81283,81293,81299,81307,81331,81343,81349,
81353,81359,81371,81373,81401,81409,81421,81439,
81457,81463,81509,81517,81527,81533,81547,81551,
81553,81559,81563,81569,81611,81619,81629,81637,
81647,81649,81667,81671,81677,81689,81701,81703,
81707,81727,81737,81749,81761,81769,81773,81799,
81817,81839,81847,81853,81869,81883,81899,81901,
81919,81929,81931,81937,81943,81953,81967,81971,
81973,82003,82007,82009,82013,82021,82031,82037,
82039,82051,82067,82073,82129,82139,82141,82153,
82163,82171,82183,82189,82193,82207,82217,82219,
82223,82231,82237,82241,82261,82267,82279,82301,
82307,82339,82349,82351,82361,82373,82387,82393,
82421,82457,82463,82469,82471,82483,82487,82493,
82499,82507,82529,82531,82549,82559,82561,82567,
82571,82591,82601,82609,82613,82619,82633,82651,
82657,82699,82721,82723,82727,82729,82757,82759,
82763,82781,82787,82793,82799,82811,82813,82837,
82847,82883,82889,82891,82903,82913,82939,82963,
82981,82997,83003,83009,83023,83047,83059,83063,
83071,83077,83089,83093,83101,83117,83137,83177,
83203,83207,83219,83221,83227,83231,83233,83243,
83257,83267,83269,83273,83299,83311,83339,83341,
83357,83383,83389,83399,83401,83407,83417,83423,
83431,83437,83443,83449,83459,83471,83477,83497,
83537,83557,83561,83563,83579,83591,83597,83609,
83617,83621,83639,83641,83653,83663,83689,83701,
83717,83719,83737,83761,83773,83777,83791,83813,
83833,83843,83857,83869,83873,83891,83903,83911,
83921,83933,83939,83969,83983,83987,84011,84017,
84047,84053,84059,84061,84067,84089,84121,84127,
84131,84137,84143,84163,84179,84181,84191,84199,
84211,84221,84223,84229,84239,84247,84263,84299,
84307,84313,84317,84319,84347,84349,84377,84389,
84391,84401,84407,84421,84431,84437,84443,84449,
84457,84463,84467,84481,84499,84503,84509,84521,
84523,84533,84551,84559,84589,84629,84631,84649,
84653,84659,84673,84691,84697,84701,84713,84719,
84731,84737,84751,84761,84787,84793,84809,84811,
84827,84857,84859,84869,84871,84913,84919,84947,
84961,84967,84977,84979,84991,85009,85021,85027,
85037,85049,85061,85081,85087,85091,85093,85103,
85109,85121,85133,85147,85159,85193,85199,85201,
85213,85223,85229,85237,85243,85247,85259,85297,
85303,85313,85331,85333,85361,85363,85369,85381,
85411,85427,85429,85439,85447,85451,85453,85469,
85487,85513,85517,85523,85531,85549,85571,85577,
85597,85601,85607,85619,85621,85627,85639,85643,
85661,85667,85669,85691,85703,85711,85717,85733,
85751,85781,85793,85817,85819,85829,85831,85837,
85843,85847,85853,85889,85903,85909,85931,85933,
85991,85999,86011,86017,86027,86029,86069,86077,
86083,86111,86113,86117,86131,86137,86143,86161,
86171,86179,86183,86197,86201,86209,86239,86243,
86249,86257,86263,86269,86287,86291,86293,86297,
86311,86323,86341,86351,86353,86357,86369,86371,
86381,86389,86399,86413,86423,86441,86453,86461,
86467,86477,86491,86501,86509,86531,86533,86539,
86561,86573,86579,86587,86599,86627,86629,86677,
86689,86693,86711,86719,86729,86743,86753,86767,
86771,86783,86813,86837,86843,86851,86857,86861,
86869,86923,86927,86929,86939,86951,86959,86969,
86981,86993,87011,87013,87037,87041,87049,87071,
87083,87103,87107,87119,87121,87133,87149,87151,
87179,87181,87187,87211,87221,87223,87251,87253,
87257,87277,87281,87293,87299,87313,87317,87323,
87337,87359,87383,87403,87407,87421,87427,87433,
87443,87473,87481,87491,87509,87511,87517,87523,
87539,87541,87547,87553,87557,87559,87583,87587,
87589,87613,87623,87629,87631,87641,87643,87649,
87671,87679,87683,87691,87697,87701,87719,87721,
87739,87743,87751,87767,87793,87797,87803,87811,
87833,87853,87869,87877,87881,87887,87911,87917,
87931,87943,87959,87961,87973,87977,87991,88001,
88003,88007,88019,88037,88069,88079,88093,88117,
88129,88169,88177,88211,88223,88237,88241,88259,
88261,88289,88301,88321,88327,88337,88339,88379,
88397,88411,88423,88427,88463,88469,88471,88493,
88499,88513,88523,88547,88589,88591,88607,88609,
88643,88651,88657,88661,88663,88667,88681,88721,
88729,88741,88747,88771,88789,88793,88799,88801,
88807,88811,88813,88817,88819,88843,88853,88861,
88867,88873,88883,88897,88903,88919,88937,88951,
88969,88993,88997,89003,89009,89017,89021,89041,
89051,89057,89069,89071,89083,89087,89101,89107,
89113,89119,89123,89137,89153,89189,89203,89209,
89213,89227,89231,89237,89261,89269,89273,89293,
89303,89317,89329,89363,89371,89381,89387,89393,
89399,89413,89417,89431,89443,89449,89459,89477,
89491,89501,89513,89519,89521,89527,89533,89561,
89563,89567,89591,89597,89599,89603,89611,89627,
89633,89653,89657,89659,89669,89671,89681,89689,
89753,89759,89767,89779,89783,89797,89809,89819,
89821,89833,89839,89849,89867,89891,89897,89899,
89909,89917,89923,89939,89959,89963,89977,89983,
89989,90001,90007,90011,90017,90019,90023,90031,
90053,90059,90067,90071,90073,90089,90107,90121,
90127,90149,90163,90173,90187,90191,90197,90199,
90203,90217,90227,90239,90247,90263,90271,90281,
90289,90313,90353,90359,90371,90373,90379,90397,
90401,90403,90407,90437,90439,90469,90473,90481,
90499,90511,90523,90527,90529,90533,90547,90583,
90599,90617,90619,90631,90641,90647,90659,90677,
90679,90697,90703,90709,90731,90749,90787,90793,
90803,90821,90823,90833,90841,90847,90863,90887,
90901,90907,90911,90917,90931,90947,90971,90977,
90989,90997,91009,91019,91033,91079,91081,91097,
91099,91121,91127,91129,91139,91141,91151,91153,
91159,91163,91183,91193,91199,91229,91237,91243,
91249,91253,91283,91291,91297,91303,91309,91331,
91367,91369,91373,91381,91387,91393,91397,91411,
91423,91433,91453,91457,91459,91463,91493,91499,
91513,91529,91541,91571,91573,91577,91583,91591,
91621,91631,91639,91673,91691,91703,91711,91733,
91753,91757,91771,91781,91801,91807,91811,91813,
91823,91837,91841,91867,91873,91909,91921,91939,
91943,91951,91957,91961,91967,91969,91997,92003,
92009,92033,92041,92051,92077,92083,92107,92111,
92119,92143,92153,92173,92177,92179,92189,92203,
92219,92221,92227,92233,92237,92243,92251,92269,
92297,92311,92317,92333,92347,92353,92357,92363,
92369,92377,92381,92383,92387,92399,92401,92413,
92419,92431,92459,92461,92467,92479,92489,92503,
92507,92551,92557,92567,92569,92581,92593,92623,
92627,92639,92641,92647,92657,92669,92671,92681,
92683,92693,92699,92707,92717,92723,92737,92753,
92761,92767,92779,92789,92791,92801,92809,92821,
92831,92849,92857,92861,92863,92867,92893,92899,
92921,92927,92941,92951,92957,92959,92987,92993,
93001,93047,93053,93059,93077,93083,93089,93097,
93103,93113,93131,93133,93139,93151,93169,93179,
93187,93199,93229,93239,93241,93251,93253,93257,
93263,93281,93283,93287,93307,93319,93323,93329,
93337,93371,93377,93383,93407,93419,93427,93463,
93479,93481,93487,93491,93493,93497,93503,93523,
93529,93553,93557,93559,93563,93581,93601,93607,
93629,93637,93683,93701,93703,93719,93739,93761,
93763,93787,93809,93811,93827,93851,93871,93887,
93889,93893,93901,93911,93913,93923,93937,93941,
93949,93967,93971,93979,93983,93997,94007,94009,
94033,94049,94057,94063,94079,94099,94109,94111,
94117,94121,94151,94153,94169,94201,94207,94219,
94229,94253,94261,94273,94291,94307,94309,94321,
94327,94331,94343,94349,94351,94379,94397,94399,
94421,94427,94433,94439,94441,94447,94463,94477,
94483,94513,94529,94531,94541,94543,94547,94559,
94561,94573,94583,94597,94603,94613,94621,94649,
94651,94687,94693,94709,94723,94727,94747,94771,
94777,94781,94789,94793,94811,94819,94823,94837,
94841,94847,94849,94873,94889,94903,94907,94933,
94949,94951,94961,94993,94999,95003,95009,95021,
95027,95063,95071,95083,95087,95089,95093,95101,
95107,95111,95131,95143,95153,95177,95189,95191,
95203,95213,95219,95231,95233,95239,95257,95261,
95267,95273,95279,95287,95311,95317,95327,95339,
95369,95383,95393,95401,95413,95419,95429,95441,
95443,95461,95467,95471,95479,95483,95507,95527,
95531,95539,95549,95561,95569,95581,95597,95603,
95617,95621,95629,95633,95651,95701,95707,95713,
95717,95723,95731,95737,95747,95773,95783,95789,
95791,95801,95803,95813,95819,95857,95869,95873,
95881,95891,95911,95917,95923,95929,95947,95957,
95959,95971,95987,95989,96001,96013,96017,96043,
96053,96059,96079,96097,96137,96149,96157,96167,
96179,96181,96199,96211,96221,96223,96233,96259,
96263,96269,96281,96289,96293,96323,96329,96331,
96337,96353,96377,96401,96419,96431,96443,96451,
96457,96461,96469,96479,96487,96493,96497,96517,
96527,96553,96557,96581,96587,96589,96601,96643,
96661,96667,96671,96697,96703,96731,96737,96739,
96749,96757,96763,96769,96779,96787,96797,96799,
96821,96823,96827,96847,96851,96857,96893,96907,
96911,96931,96953,96959,96973,96979,96989,96997,
97001,97003,97007,97021,97039,97073,97081,97103,
97117,97127,97151,97157,97159,97169,97171,97177,
97187,97213,97231,97241,97259,97283,97301,97303,
97327,97367,97369,97373,97379,97381,97387,97397,
97423,97429,97441,97453,97459,97463,97499,97501,
97511,97523,97547,97549,97553,97561,97571,97577,
97579,97583,97607,97609,97613,97649,97651,97673,
97687,97711,97729,97771,97777,97787,97789,97813,
97829,97841,97843,97847,97849,97859,97861,97871,
97879,97883,97919,97927,97931,97943,97961,97967,
97973,97987,98009,98011,98017,98041,98047,98057,
98081,98101,98123,98129,98143,98179,98207,98213,
98221,98227,98251,98257,98269,98297,98299,98317,
98321,98323,98327,98347,98369,98377,98387,98389,
98407,98411,98419,98429,98443,98453,98459,98467,
98473,98479,98491,98507,98519,98533,98543,98561,
98563,98573,98597,98621,98627,98639,98641,98663,
98669,98689,98711,98713,98717,98729,98731,98737,
98773,98779,98801,98807,98809,98837,98849,98867,
98869,98873,98887,98893,98897,98899,98909,98911,
98927,98929,98939,98947,98953,98963,98981,98993,
98999,99013,99017,99023,99041,99053,99079,99083,
99089,99103,99109,99119,99131,99133,99137,99139,
99149,99173,99181,99191,99223,99233,99241,99251,
99257,99259,99277,99289,99317,99347,99349,99367,
99371,99377,99391,99397,99401,99409,99431,99439,
99469,99487,99497,99523,99527,99529,99551,99559,
99563,99571,99577,99581,99607,99611,99623,99643,
99661,99667,99679,99689,99707,99709,99713,99719,
99721,99733,99761,99767,99787,99793,99809,99817,
99823,99829,99833,99839,99859,99871,99877,99881,
99901,99907,99923,99929,99961,99971,99989,99991,
100003,100019,100043,100049,100057,100069,100103,100109,
100129,100151,100153,100169,100183,100189,100193,100207,
100213,100237,100267,100271,100279,100291,100297,100313,
100333,100343,100357,100361,100363,100379,100391,100393,
100403,100411,100417,100447,100459,100469,100483,100493,
100501,100511,100517,100519,100523,100537,100547,100549,
100559,100591,100609,100613,100621,100649,100669,100673,
100693,100699,100703,100733,100741,100747,100769,100787,
100799,100801,100811,100823,100829,100847,100853,100907,
100913,100927,100931,100937,100943,100957,100981,100987,
100999,101009,101021,101027,101051,101063,101081,101089,
101107,101111,101113,101117,101119,101141,101149,101159,
101161,101173,101183,101197,101203,101207,101209,101221,
101267,101273,101279,101281,101287,101293,101323,101333,
101341,101347,101359,101363,101377,101383,101399,101411,
101419,101429,101449,101467,101477,101483,101489,101501,
101503,101513,101527,101531,101533,101537,101561,101573,
101581,101599,101603,101611,101627,101641,101653,101663,
101681,101693,101701,101719,101723,101737,101741,101747,
101749,101771,101789,101797,101807,101833,101837,101839,
101863,101869,101873,101879,101891,101917,101921,101929,
101939,101957,101963,101977,101987,101999,102001,102013,
102019,102023,102031,102043,102059,102061,102071,102077,
102079,102101,102103,102107,102121,102139,102149,102161,
102181,102191,102197,102199,102203,102217,102229,102233,
102241,102251,102253,102259,102293,102299,102301,102317,
102329,102337,102359,102367,102397,102407,102409,102433,
102437,102451,102461,102481,102497,102499,102503,102523,
102533,102539,102547,102551,102559,102563,102587,102593,
102607,102611,102643,102647,102653,102667,102673,102677,
102679,102701,102761,102763,102769,102793,102797,102811,
102829,102841,102859,102871,102877,102881,102911,102913,
102929,102931,102953,102967,102983,103001,103007,103043,
103049,103067,103069,103079,103087,103091,103093,103099,
103123,103141,103171,103177,103183,103217,103231,103237,
103289,103291,103307,103319,103333,103349,103357,103387,
103391,103393,103399,103409,103421,103423,103451,103457,
103471,103483,103511,103529,103549,103553,103561,103567,
103573,103577,103583,103591,103613,103619,103643,103651,
103657,103669,103681,103687,103699,103703,103723,103769,
103787,103801,103811,103813,103837,103841,103843,103867,
103889,103903,103913,103919,103951,103963,103967,103969,
103979,103981,103991,103993,103997,104003,104009,104021,
104033,104047,104053,104059,104087,104089,104107,104113,
104119,104123,104147,104149,104161,104173,104179,104183,
104207,104231,104233,104239,104243,104281,104287,104297,
104309,104311,104323,104327,104347,104369,104381,104383,
104393,104399,104417,104459,104471,104473,104479,104491,
104513,104527,104537,104543,104549,104551,104561,104579,
104593,104597,104623,104639,104651,104659,104677,104681,
104683,104693,104701,104707,104711,104717,104723,104729,
};

// tty style printing

int out_index, out_length;
char *out_str;
int char_count, last_char;

char *power_str = "^";

void
print(U *p)
{
	if (car(p) == symbol(SETQ)) {
		print_expr(cadr(p));
		printstr(" = ");
		print_expr(caddr(p));
	} else
		print_expr(p);
	print_char('\n');
}

void
print_subexpr(U *p)
{
	print_char('(');
	print_expr(p);
	print_char(')');
}

void
print_expr(U *p)
{
	if (isadd(p)) {
		p = cdr(p);
		if (sign_of_term(car(p)) == '-')
			print_str("-");
		print_term(car(p));
		p = cdr(p);
		while (iscons(p)) {
			if (sign_of_term(car(p)) == '+')
				print_str(" + ");
			else
				print_str(" - ");
			print_term(car(p));
			p = cdr(p);
		}
	} else {
		if (sign_of_term(p) == '-')
			print_str("-");
		print_term(p);
	}
}

int
sign_of_term(U *p)
{
	if (car(p) == symbol(MULTIPLY) && isnum(cadr(p)) && lessp(cadr(p), zero))
		return '-';
	else if (isnum(p) && lessp(p, zero))
		return '-';
	else
		return '+';
}

#undef A
#undef B

#define A p3
#define B p4

void
print_a_over_b(U *p)
{
	int flag, n, d;
	U *p1, *p2;
	save();
	// count numerators and denominators
	n = 0;
	d = 0;
	p1 = cdr(p);
	p2 = car(p1);
	if (isrational(p2)) {
		push(p2);
		mp_numerator();
		absval();
		A = pop();
		push(p2);
		mp_denominator();
		B = pop();
		if (!isplusone(A))
			n++;
		if (!isplusone(B))
			d++;
		p1 = cdr(p1);
	} else {
		A = one;
		B = one;
	}
	while (iscons(p1)) {
		p2 = car(p1);
		if (is_denominator(p2))
			d++;
		else
			n++;
		p1 = cdr(p1);
	}
	if (n == 0)
		print_char('1');
	else {
		flag = 0;
		p1 = cdr(p);
		if (isrational(car(p1)))
			p1 = cdr(p1);
		if (!isplusone(A)) {
			print_factor(A);
			flag = 1;
		}
		while (iscons(p1)) {
			p2 = car(p1);
			if (is_denominator(p2))
				;
			else {
				if (flag)
					print_multiply_sign();
				print_factor(p2);
				flag = 1;
			}
			p1 = cdr(p1);
		}
	}
	print_str(" / ");
	if (d > 1)
		print_char('(');
	flag = 0;
	p1 = cdr(p);
	if (isrational(car(p1)))
		p1 = cdr(p1);
	if (!isplusone(B)) {
		print_factor(B);
		flag = 1;
	}
	while (iscons(p1)) {
		p2 = car(p1);
		if (is_denominator(p2)) {
			if (flag)
				print_multiply_sign();
			print_denom(p2, d);
			flag = 1;
		}
		p1 = cdr(p1);
	}
	if (d > 1)
		print_char(')');
	restore();
}

void
print_term(U *p)
{
	if (car(p) == symbol(MULTIPLY) && any_denominators(p)) {
		print_a_over_b(p);
		return;
	}
	if (car(p) == symbol(MULTIPLY)) {
		p = cdr(p);
		// coeff -1?
		if (isminusone(car(p))) {
//			print_char('-');
			p = cdr(p);
		}
		print_factor(car(p));
		p = cdr(p);
		while (iscons(p)) {
			print_multiply_sign();
			print_factor(car(p));
			p = cdr(p);
		}
	} else
		print_factor(p);
}

// prints stuff after the divide symbol "/"

// d is the number of denominators

#undef BASE
#undef EXPO

#define BASE p1
#define EXPO p2

void
print_denom(U *p, int d)
{
	save();
	BASE = cadr(p);
	EXPO = caddr(p);
	// i.e. 1 / (2^(1/3))
	if (d == 1 && !isminusone(EXPO))
		print_char('(');
	if (isfraction(BASE) || car(BASE) == symbol(ADD) || car(BASE) == symbol(MULTIPLY) || car(BASE) == symbol(POWER) || lessp(BASE, zero)) {
			print_char('(');
			print_expr(BASE);
			print_char(')');
	} else
		print_expr(BASE);
	if (isminusone(EXPO)) {
		restore();
		return;
	}
	print_str(power_str);
	push(EXPO);
	negate();
	EXPO = pop();
	if (isfraction(EXPO) || car(EXPO) == symbol(ADD) || car(EXPO) == symbol(MULTIPLY) || car(EXPO) == symbol(POWER)) {
		print_char('(');
		print_expr(EXPO);
		print_char(')');
	} else
		print_expr(EXPO);
	if (d == 1)
		print_char(')');
	restore();
}

void
print_factor(U *p)
{
	if (isnum(p)) {
		print_number(p);
		return;
	}
	if (isstr(p)) {
		//print_str("\"");
		print_str(p->u.str);
		//print_str("\"");
		return;
	}
	if (istensor(p)) {
		print_tensor(p);
		return;
	}
	if (isadd(p) || car(p) == symbol(MULTIPLY)) {
		print_str("(");
		print_expr(p);
		print_str(")");
		return;
	}
	if (car(p) == symbol(POWER)) {
		if (cadr(p) == symbol(EXP1)) {
			print_str("exp(");
			print_expr(caddr(p));
			print_str(")");
			return;
		}
		if (isminusone(caddr(p))) {
			print_str("1 / ");
			if (iscons(cadr(p))) {
				print_str("(");
				print_expr(cadr(p));
				print_str(")");
			} else
				print_expr(cadr(p));
			return;
		}
		if (isadd(cadr(p)) || caadr(p) == symbol(MULTIPLY) || caadr(p) == symbol(POWER) || isnegativenumber(cadr(p))) {
			print_str("(");
			print_expr(cadr(p));
			print_str(")");
		} else if (isnum(cadr(p)) && (lessp(cadr(p), zero) || isfraction(cadr(p)))) {
			print_str("(");
			print_factor(cadr(p));
			print_str(")");
		} else
			print_factor(cadr(p));
		print_str(power_str);
		if (iscons(caddr(p)) || isfraction(caddr(p)) || (isnum(caddr(p)) && lessp(caddr(p), zero))) {
			print_str("(");
			print_expr(caddr(p));
			print_str(")");
		} else
			print_factor(caddr(p));
		return;
	}
//	if (car(p) == _list) {
//		print_str("{");
//		p = cdr(p);
//		if (iscons(p)) {
//			print_expr(car(p));
//			p = cdr(p);
//		}
//		while (iscons(p)) {
//			print_str(",");
//			print_expr(car(p));
//			p = cdr(p);
//		}
//		print_str("}");
//		return;
//	}
	if (car(p) == symbol(INDEX) && issymbol(cadr(p))) {
		print_index_function(p);
		return;
	}
	if (car(p) == symbol(FACTORIAL)) {
		print_factorial_function(p);
		return;
	}
	if (iscons(p)) {
		//if (car(p) == symbol(FORMAL) && cadr(p)->k == SYM) {
		//	print_str(((struct symbol *) cadr(p))->name);
		//	return;
		//}
		print_factor(car(p));
		p = cdr(p);
		print_str("(");
		if (iscons(p)) {
			print_expr(car(p));
			p = cdr(p);
			while (iscons(p)) {
				print_str(",");
				print_expr(car(p));
				p = cdr(p);
			}
		}
		print_str(")");
		return;
	}
	if (p == symbol(DERIVATIVE))
		print_char('d');
	else if (p == symbol(EXP1))
		print_str("exp(1)");
	else if (p == symbol(PI))
		print_str("pi");
	else
		print_str(get_printname(p));
}

void
print_index_function(U *p)
{
	p = cdr(p);
	if (caar(p) == symbol(ADD) || caar(p) == symbol(MULTIPLY) || caar(p) == symbol(POWER) || caar(p) == symbol(FACTORIAL))
		print_subexpr(car(p));
	else
		print_expr(car(p));
	print_char('[');
	p = cdr(p);
	if (iscons(p)) {
		print_expr(car(p));
		p = cdr(p);
		while(iscons(p)) {
			print_char(',');
			print_expr(car(p));
			p = cdr(p);
		}
	}
	print_char(']');
}

void
print_factorial_function(U *p)
{
	p = cadr(p);
	if (car(p) == symbol(ADD) || car(p) == symbol(MULTIPLY) || car(p) == symbol(POWER) || car(p) == symbol(FACTORIAL))
		print_subexpr(p);
	else
		print_expr(p);
	print_char('!');
}

void
print_tensor(U *p)
{
	int k = 0;
	print_tensor_inner(p, 0, &k);
}

void
print_tensor_inner(U *p, int j, int *k)
{
	int i;
	print_str("(");
	for (i = 0; i < p->u.tensor->dim[j]; i++) {
		if (j + 1 == p->u.tensor->ndim) {
			print_expr(p->u.tensor->elem[*k]);
			*k = *k + 1;
		} else
			print_tensor_inner(p, j + 1, k);
		if (i + 1 < p->u.tensor->dim[j]) {
			print_str(",");
		}
	}
	print_str(")");
}

void
print_str(char *s)
{
	while (*s)
		print_char(*s++);
}

void
print_char(int c)
{
	last_char = c;
	char_count++;
//	if (display_flag == 1)
//		displaychar(c);
//	else
		printchar(c);
}

void
print_function_definition(U *p)
{
	print_str(get_printname(p));
	print_arg_list(cadr(get_binding(p)));
	print_str("=");
	print_expr(caddr(get_binding(p)));
	print_str("\n");
}

void
print_arg_list(U *p)
{
	print_str("(");
	if (iscons(p)) {
		print_str(get_printname(car(p)));
		p = cdr(p);
		while (iscons(p)) {
			print_str(",");
			print_str(get_printname(car(p)));
			p = cdr(p);
		}
	}
	print_str(")");
}

void
print_lisp(U *p)
{
	print1(p);
	print_str("\n");
}

void
print1(U *p)
{
	switch (p->k) {
	case CONS:
		print_str("(");
		print1(car(p));
		p = cdr(p);
		while (iscons(p)) {
			print_str(" ");
			print1(car(p));
			p = cdr(p);
		}
		if (p != symbol(NIL)) {
			print_str(" . ");
			print1(p);
		}
		print_str(")");
		break;
	case STR:
		//print_str("\"");
		print_str(p->u.str);
		//print_str("\"");
		break;
	case NUM:
	case DOUBLE:
		print_number(p);
		break;
	case SYM:
		print_str(get_printname(p));
		break;
	default:
		print_str("<tensor>");
		break;
	}
}

void
print_multiply_sign(void)
{
	print_str(" ");
}

int
is_denominator(U *p)
{
	if (car(p) == symbol(POWER) && cadr(p) != symbol(EXP1) && isnegativeterm(caddr(p)))
		return 1;
	else
		return 0;
}

// don't consider the leading fraction

// we want 2/3*a*b*c instead of 2*a*b*c/3

int
any_denominators(U *p)
{
	U *q;
	p = cdr(p);
//	if (isfraction(car(p)))
//		return 1;
	while (iscons(p)) {
		q = car(p);
		if (is_denominator(q))
			return 1;
		p = cdr(p);
	}
	return 0;
}

// 'product' function

#undef I
#undef X

#define I p5
#define X p6

void
eval_product(void)
{
	int i, j, k;
	// 1st arg (quoted)
	X = cadr(p1);
	if (!issymbol(X))
		stop("product: 1st arg?");
	// 2nd arg
	push(caddr(p1));
	eval();
	j = pop_integer();
	if (j == (int) 0x80000000)
		stop("product: 2nd arg?");
	// 3rd arg
	push(cadddr(p1));
	eval();
	k = pop_integer();
	if (k == (int) 0x80000000)
		stop("product: 3rd arg?");
	// 4th arg
	p1 = caddddr(p1);
	push_binding(X);
	push_integer(1);
	for (i = j; i <= k; i++) {
		push_integer(i);
		I = pop();
		set_binding(X, I);
		push(p1);
		eval();
		multiply();
	}
	p1 = pop();
	pop_binding(X);
	push(p1);
}

//	Add rational numbers
//
//	Input:		tos-2		addend
//
//			tos-1		addend
//
//	Output:		sum on stack

void
qadd(void)
{
	unsigned int *a, *ab, *b, *ba, *c;
	save();
	p2 = pop();
	p1 = pop();
	ab = mmul(p1->u.q.a, p2->u.q.b);
	ba = mmul(p1->u.q.b, p2->u.q.a);
	a = madd(ab, ba);
	mfree(ab);
	mfree(ba);
	// zero?
	if (MZERO(a)) {
		mfree(a);
		push(zero);
		restore();
		return;
	}
	b = mmul(p1->u.q.b, p2->u.q.b);
	c = mgcd(a, b);
	MSIGN(c) = MSIGN(b);
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = mdiv(a, c);
	p1->u.q.b = mdiv(b, c);
	mfree(a);
	mfree(b);
	mfree(c);
	push(p1);
	restore();
}

//	Divide rational numbers
//
//	Input:		tos-2		dividend
//
//			tos-1		divisor
//
//	Output:		quotient on stack

void
qdiv(void)
{
	unsigned int *aa, *bb, *c;
	save();
	p2 = pop();
	p1 = pop();
	// zero?
	if (MZERO(p2->u.q.a))
		stop("divide by zero");
	if (MZERO(p1->u.q.a)) {
		push(zero);
		restore();
		return;
	}
	aa = mmul(p1->u.q.a, p2->u.q.b);
	bb = mmul(p1->u.q.b, p2->u.q.a);
	c = mgcd(aa, bb);
	MSIGN(c) = MSIGN(bb);
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = mdiv(aa, c);
	p1->u.q.b = mdiv(bb, c);
	mfree(aa);
	mfree(bb);
	mfree(c);
	push(p1);
	restore();
}

//	Multiply rational numbers
//
//	Input:		tos-2		multiplicand
//
//			tos-1		multiplier
//
//	Output:		product on stack

void
qmul(void)
{
	unsigned int *aa, *bb, *c;
	save();
	p2 = pop();
	p1 = pop();
	// zero?
	if (MZERO(p1->u.q.a) || MZERO(p2->u.q.a)) {
		push(zero);
		restore();
		return;
	}
	aa = mmul(p1->u.q.a, p2->u.q.a);
	bb = mmul(p1->u.q.b, p2->u.q.b);
	c = mgcd(aa, bb);
	MSIGN(c) = MSIGN(bb);
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = mdiv(aa, c);
	p1->u.q.b = mdiv(bb, c);
	mfree(aa);
	mfree(bb);
	mfree(c);
	push(p1);
	restore();
}

// Rational power function

#undef BASE
#undef EXPO

#define BASE p1
#define EXPO p2

void
qpow(void)
{
	save();
	qpowf();
	restore();
}

void
qpowf(void)
{
	int expo;
	unsigned int a, b, *t, *x, *y;
	EXPO = pop();
	BASE = pop();
	// if base is 1 or exponent is 0 then return 1
	if (isplusone(BASE) || iszero(EXPO)) {
		push_integer(1);
		return;
	}
	// if base is zero then return 0
	if (iszero(BASE)) {
		if (isnegativenumber(EXPO))
			stop("divide by zero");
		push(zero);
		return;
	}
	// if exponent is 1 then return base
	if (isplusone(EXPO)) {
		push(BASE);
		return;
	}
	// if exponent is integer then power
	if (isinteger(EXPO)) {
		push(EXPO);
		expo = pop_integer();
		if (expo == (int) 0x80000000) {
			// expo greater than 32 bits
			push_symbol(POWER);
			push(BASE);
			push(EXPO);
			list(3);
			return;
		}
		x = mpow(BASE->u.q.a, abs(expo));
		y = mpow(BASE->u.q.b, abs(expo));
		if (expo < 0) {
			t = x;
			x = y;
			y = t;
			MSIGN(x) = MSIGN(y);
			MSIGN(y) = 1;
		}
		p3 = alloc();
		p3->k = NUM;
		p3->u.q.a = x;
		p3->u.q.b = y;
		push(p3);
		return;
	}
	// from here on out the exponent is NOT an integer
	// if base is -1 then normalize polar angle
	if (isminusone(BASE)) {
		push(EXPO);
		normalize_angle();
		return;
	}
	// if base is negative then (-N)^M -> N^M * (-1)^M
	if (isnegativenumber(BASE)) {
		push(BASE);
		negate();
		push(EXPO);
		qpow();
		push_integer(-1);
		push(EXPO);
		qpow();
		multiply();
		return;
	}
	// if BASE is not an integer then power numerator and denominator
	if (!isinteger(BASE)) {
		push(BASE);
		mp_numerator();
		push(EXPO);
		qpow();
		push(BASE);
		mp_denominator();
		push(EXPO);
		negate();
		qpow();
		multiply();
		return;
	}
	// At this point BASE is a positive integer.
	// If BASE is small then factor it.
	if (is_small_integer(BASE)) {
		push(BASE);
		push(EXPO);
		quickfactor();
		return;
	}
	// At this point BASE is a positive integer and EXPO is not an integer.
	if (MLENGTH(EXPO->u.q.a) > 1 || MLENGTH(EXPO->u.q.b) > 1) {
		push_symbol(POWER);
		push(BASE);
		push(EXPO);
		list(3);
		return;
	}
	a = EXPO->u.q.a[0];
	b = EXPO->u.q.b[0];
	x = mroot(BASE->u.q.a, b);
	if (x == 0) {
		push_symbol(POWER);
		push(BASE);
		push(EXPO);
		list(3);
		return;
	}
	y = mpow(x, a);
	mfree(x);
	p3 = alloc();
	p3->k = NUM;
	if (MSIGN(EXPO->u.q.a) == -1) {
		p3->u.q.a = mint(1);
		p3->u.q.b = y;
	} else {
		p3->u.q.a = y;
		p3->u.q.b = mint(1);
	}
	push(p3);
}

//-----------------------------------------------------------------------------
//
//	Normalize the angle of unit imaginary, i.e. (-1) ^ N
//
//	Input:		N on stack (must be rational, not float)
//
//	Output:		Result on stack
//
//	Note:
//
//	n = q * d + r
//
//	Example:
//						n	d	q	r
//
//	(-1)^(8/3)	->	 (-1)^(2/3)	8	3	2	2
//	(-1)^(7/3)	->	 (-1)^(1/3)	7	3	2	1
//	(-1)^(5/3)	->	-(-1)^(2/3)	5	3	1	2
//	(-1)^(4/3)	->	-(-1)^(1/3)	4	3	1	1
//	(-1)^(2/3)	->	 (-1)^(2/3)	2	3	0	2
//	(-1)^(1/3)	->	 (-1)^(1/3)	1	3	0	1
//
//	(-1)^(-1/3)	->	-(-1)^(2/3)	-1	3	-1	2
//	(-1)^(-2/3)	->	-(-1)^(1/3)	-2	3	-1	1
//	(-1)^(-4/3)	->	 (-1)^(2/3)	-4	3	-2	2
//	(-1)^(-5/3)	->	 (-1)^(1/3)	-5	3	-2	1
//	(-1)^(-7/3)	->	-(-1)^(2/3)	-7	3	-3	2
//	(-1)^(-8/3)	->	-(-1)^(1/3)	-8	3	-3	1
//
//-----------------------------------------------------------------------------

#undef A
#undef Q
#undef R

#define A p1
#define Q p2
#define R p3

void
normalize_angle(void)
{
	save();
	A = pop();
	// integer exponent?
	if (isinteger(A)) {
		if (A->u.q.a[0] & 1)
			push_integer(-1); // odd exponent
		else
			push_integer(1); // even exponent
		restore();
		return;
	}
	// floor
	push(A);
	bignum_truncate();
	Q = pop();
	if (isnegativenumber(A)) {
		push(Q);
		push_integer(-1);
		add();
		Q = pop();
	}
	// remainder (always positive)
	push(A);
	push(Q);
	subtract();
	R = pop();
	// remainder becomes new angle
	push_symbol(POWER);
	push_integer(-1);
	push(R);
	list(3);
	// negate if quotient is odd
	if (Q->u.q.a[0] & 1)
		negate();
	restore();
}

int
is_small_integer(U *p)
{
	if (isinteger(p) && MLENGTH(p->u.q.a) == 1 && (p->u.q.a[0] & 0x80000000) == 0)
		return 1;
	else
		return 0;
}

//	Subtract rational numbers
//
//	Input:		tos-2		minuend
//
//			tos-1		subtrahend
//
//	Output:		difference on stack

void
qsub(void)
{
	unsigned int *a, *ab, *b, *ba, *c;
	save();
	p2 = pop();
	p1 = pop();
	ab = mmul(p1->u.q.a, p2->u.q.b);
	ba = mmul(p1->u.q.b, p2->u.q.a);
	a = msub(ab, ba);
	mfree(ab);
	mfree(ba);
	// zero?
	if (MZERO(a)) {
		mfree(a);
		push(zero);
		restore();
		return;
	}
	b = mmul(p1->u.q.b, p2->u.q.b);
	c = mgcd(a, b);
	MSIGN(c) = MSIGN(b);
	p1 = alloc();
	p1->k = NUM;
	p1->u.q.a = mdiv(a, c);
	p1->u.q.b = mdiv(b, c);
	mfree(a);
	mfree(b);
	mfree(c);
	push(p1);
	restore();
}

//-----------------------------------------------------------------------------
//
//	Factor small numerical powers
//
//	Input:		tos-2		Base (positive integer < 2^31 - 1)
//
//			tos-1		Exponent
//
//	Output:		Expr on stack
//
//-----------------------------------------------------------------------------

#undef BASE
#undef EXPO

#define BASE p1
#define EXPO p2

void
quickfactor(void)
{
	int h, i, n;
	U **s;
	save();
	EXPO = pop();
	BASE = pop();
	h = tos;
	push(BASE);
	factor_small_number();
	n = tos - h;
	s = stack + h;
	for (i = 0; i < n; i += 2) {
		push(s[i]);		// factored base
		push(s[i + 1]);		// factored exponent
		push(EXPO);
		multiply();
		quickpower();
	}
	// stack has n results from factor_number_raw()
	// on top of that are all the expressions from quickpower()
	// multiply the quickpower() results
	multiply_all(tos - h - n);
	p1 = pop();
	tos = h;
	push(p1);
	restore();
}

// BASE is a prime number so power is simpler

void
quickpower(void)
{
	int expo;
	save();
	EXPO = pop();
	BASE = pop();
	push(EXPO);
	bignum_truncate();
	p3 = pop();
	push(EXPO);
	push(p3);
	subtract();
	p4 = pop();
	// fractional part of EXPO
	if (!iszero(p4)) {
		push_symbol(POWER);
		push(BASE);
		push(p4);
		list(3);
	}
	push(p3);
	expo = pop_integer();
	if (expo == (int) 0x80000000) {
		push_symbol(POWER);
		push(BASE);
		push(p3);
		list(3);
		restore();
		return;
	}
	if (expo == 0) {
		restore();
		return;
	}
	push(BASE);
	bignum_power_number(expo);
	restore();
}

// Divide polynomials

void
eval_quotient(void)
{
	push(cadr(p1));			// 1st arg, p(x)
	eval();
	push(caddr(p1));		// 2nd arg, q(x)
	eval();
	push(cadddr(p1));		// 3rd arg, x
	eval();
	p1 = pop();			// default x
	if (p1 == symbol(NIL))
		p1 = symbol(SYMBOL_X);
	push(p1);
	divpoly();
}

//-----------------------------------------------------------------------------
//
//	Divide polynomials
//
//	Input:		tos-3		Dividend
//
//			tos-2		Divisor
//
//			tos-1		x
//
//	Output:		tos-1		Quotient
//
//-----------------------------------------------------------------------------

#undef DIVIDEND
#undef DIVISOR
#undef X
#undef Q
#undef QUOTIENT

#define DIVIDEND p1
#define DIVISOR p2
#define X p3
#define Q p4
#define QUOTIENT p5

void
divpoly(void)
{
	int h, i, m, n, x;
	U **dividend, **divisor;
	save();
	X = pop();
	DIVISOR = pop();
	DIVIDEND = pop();
	h = tos;
	dividend = stack + tos;
	push(DIVIDEND);
	push(X);
	m = coeff() - 1;	// m is dividend's power
	divisor = stack + tos;
	push(DIVISOR);
	push(X);
	n = coeff() - 1;	// n is divisor's power
	x = m - n;
	push_integer(0);
	QUOTIENT = pop();
	while (x >= 0) {
		push(dividend[m]);
		push(divisor[n]);
		divide();
		Q = pop();
		for (i = 0; i <= n; i++) {
			push(dividend[x + i]);
			push(divisor[i]);
			push(Q);
			multiply();
			subtract();
			dividend[x + i] = pop();
		}
		push(QUOTIENT);
		push(Q);
		push(X);
		push_integer(x);
		power();
		multiply();
		add();
		QUOTIENT = pop();
		m--;
		x--;
	}
	tos = h;
	push(QUOTIENT);
	restore();
}

#define YYDEBUG 0

void
eval_rationalize(void)
{
	push(cadr(p1));
	eval();
	rationalize();
}

void
rationalize(void)
{
	int x = expanding;
	save();
	yyrationalize();
	restore();
	expanding = x;
}

void
yyrationalize(void)
{
	p1 = pop();
	if (istensor(p1)) {
		rationalize_tensor();
		return;
	}
	expanding = 0;
	if (car(p1) != symbol(ADD)) {
		push(p1);
		return;
	}
#if YYDEBUG
	printf("rationalize: this is the input expr:\n");
	print(p1);
#endif
	// get common denominator
	push(one);
	multiply_denominators(p1);
	p2 = pop();
#if YYDEBUG
	printf("rationalize: this is the common denominator:\n");
	print(p2);
#endif
	// multiply each term by common denominator
	push(zero);
	p3 = cdr(p1);
	while (iscons(p3)) {
		push(p2);
		push(car(p3));
		multiply();
		add();
		p3 = cdr(p3);
	}
#if YYDEBUG
	printf("rationalize: original expr times common denominator:\n");
	print(stack[tos - 1]);
#endif
	// collect common factors
	Condense();
#if YYDEBUG
	printf("rationalize: after factoring:\n");
	print(stack[tos - 1]);
#endif
	// divide by common denominator
	push(p2);
	divide();
#if YYDEBUG
	printf("rationalize: after dividing by common denom. (and we're done):\n");
	print(stack[tos - 1]);
#endif
}

void
multiply_denominators(U *p)
{
	if (car(p) == symbol(ADD)) {
		p = cdr(p);
		while (iscons(p)) {
			multiply_denominators_term(car(p));
			p = cdr(p);
		}
	} else
		multiply_denominators_term(p);
}

void
multiply_denominators_term(U *p)
{
	if (car(p) == symbol(MULTIPLY)) {
		p = cdr(p);
		while (iscons(p)) {
			multiply_denominators_factor(car(p));
			p = cdr(p);
		}
	} else
		multiply_denominators_factor(p);
}

void
multiply_denominators_factor(U *p)
{
	if (car(p) != symbol(POWER))
		return;
	push(p);
	p = caddr(p);
	// like x^(-2) ?
	if (isnegativenumber(p)) {
		inverse();
		rationalize_lcm();
		return;
	}
	// like x^(-a) ?
	if (car(p) == symbol(MULTIPLY) && isnegativenumber(cadr(p))) {
		inverse();
		rationalize_lcm();
		return;
	}
	// no match
	pop();
}

void
rationalize_tensor(void)
{
	int i, n;
	push(p1);
	eval(); // makes a copy
	p1 = pop();
	if (!istensor(p1)) { // might be zero
		push(p1);
		return;
	}
	n = p1->u.tensor->nelem;
	for (i = 0; i < n; i++) {
		push(p1->u.tensor->elem[i]);
		rationalize();
		p1->u.tensor->elem[i] = pop();
	}
	push(p1);
}

void
rationalize_lcm(void)
{
	save();
	p1 = pop();
	p2 = pop();
	push(p1);
	push(p2);
	multiply();
	push(p1);
	push(p2);
	gcd();
	divide();
	restore();
}

/* Returns the real part of complex z

	z		real(z)
	-		-------

	a + i b		a

	exp(i a)	cos(a)
*/

void
eval_real(void)
{
	push(cadr(p1));
	eval();
	real();
}

void
real(void)
{
	save();
	rect();
	p1 = pop();
	push(p1);
	push(p1);
	conjugate();
	add();
	push_integer(2);
	divide();
	restore();
}

/* Convert complex z to rectangular form

	Input:		push	z

	Output:		Result on stack
*/

void
eval_rect(void)
{
	push(cadr(p1));
	eval();
	rect();
}

void
rect(void)
{
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD)) {
		push_integer(0);
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			rect();
			add();
			p1 = cdr(p1);
		}
	} else {
		push(p1);	// mag(z) * (cos(arg(z)) + i sin(arg(z)))
		mag();
		push(p1);
		arg();
		p1 = pop();
		push(p1);
		cosine();
		push(imaginaryunit);
		push(p1);
		sine();
		multiply();
		add();
		multiply();
	}
	restore();
}

// Rewrite by expanding all symbols

void
rewrite(void)
{
	int h;
	save();
	p1 = pop();
	if (istensor(p1)) {
		rewrite_tensor();
		restore();
		return;
	}
	if (iscons(p1)) {
		h = tos;
		push(car(p1)); // Do not rewrite function name
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			rewrite();
			p1 = cdr(p1);
		}
		list(tos - h);
		restore();
		return;
	}
	// If not a symbol then done
	if (!issymbol(p1)) {
		push(p1);
		restore();
		return;
	}
	// Get the symbol's binding, try again
	p2 = get_binding(p1);
	push(p2);
	if (p1 != p2)
		rewrite();
	restore();
}

void
rewrite_tensor(void)
{
	int i;
	push(p1);
	copy_tensor();
	p1 = pop();
	for (i = 0; i < p1->u.tensor->nelem; i++) {
		push(p1->u.tensor->elem[i]);
		rewrite();
		p1->u.tensor->elem[i] = pop();
	}
	push(p1);
}

#undef POLY
#undef X
#undef A
#undef B
#undef C
#undef Y

#define POLY p1
#define X p2
#define A p3
#define B p4
#define C p5
#define Y p6

void
eval_roots(void)
{
	// A == B -> A - B
	p2 = cadr(p1);
	if (car(p2) == symbol(SETQ) || car(p2) == symbol(TESTEQ)) {
		push(cadr(p2));
		eval();
		push(caddr(p2));
		eval();
		subtract();
	} else {
		push(p2);
		eval();
		p2 = pop();
		if (car(p2) == symbol(SETQ) || car(p2) == symbol(TESTEQ)) {
			push(cadr(p2));
			eval();
			push(caddr(p2));
			eval();
			subtract();
		} else
			push(p2);
	}
	// 2nd arg, x
	push(caddr(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		guess();
	else
		push(p2);
	p2 = pop();
	p1 = pop();
	if (!ispoly(p1, p2))
		stop("roots: 1st argument is not a polynomial");
	push(p1);
	push(p2);
	roots();
}

void
roots(void)
{
	int h, i, n;
	h = tos - 2;
	roots2();
	n = tos - h;
	if (n == 0)
		stop("roots: the polynomial is not factorable, try nroots");
	if (n == 1)
		return;
	sort_stack(n);
	save();
	p1 = alloc_tensor(n);
	p1->u.tensor->ndim = 1;
	p1->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++)
		p1->u.tensor->elem[i] = stack[h + i];
	tos = h;
	push(p1);
	restore();
}

void
roots2(void)
{
	save();
	p2 = pop();
	p1 = pop();
	push(p1);
	push(p2);
	factorpoly();
	p1 = pop();
	if (car(p1) == symbol(MULTIPLY)) {
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			push(p2);
			roots3();
			p1 = cdr(p1);
		}
	} else {
		push(p1);
		push(p2);
		roots3();
	}
	restore();
}

void
roots3(void)
{
	save();
	p2 = pop();
	p1 = pop();
	if (car(p1) == symbol(POWER) && ispoly(cadr(p1), p2) && isposint(caddr(p1))) {
		push(cadr(p1));
		push(p2);
		mini_solve();
	} else if (ispoly(p1, p2)) {
		push(p1);
		push(p2);
		mini_solve();
	}
	restore();
}

//-----------------------------------------------------------------------------
//
//	Input:		stack[tos - 2]		polynomial
//
//			stack[tos - 1]		dependent symbol
//
//	Output:		stack			roots on stack
//
//						(input args are popped first)
//
//-----------------------------------------------------------------------------

void
mini_solve(void)
{
	int n;
	save();
	X = pop();
	POLY = pop();
	push(POLY);
	push(X);
	n = coeff();
	// AX + B, X = -B/A
	if (n == 2) {
		A = pop();
		B = pop();
		push(B);
		push(A);
		divide();
		negate();
		restore();
		return;
	}
	// AX^2 + BX + C, X = (-B +/- (B^2 - 4AC)^(1/2)) / (2A)
	if (n == 3) {
		A = pop();
		B = pop();
		C = pop();
		push(B);
		push(B);
		multiply();
		push_integer(4);
		push(A);
		multiply();
		push(C);
		multiply();
		subtract();
		push_rational(1, 2);
		power();
		Y = pop();
		push(Y);			// 1st root
		push(B);
		subtract();
		push(A);
		divide();
		push_rational(1, 2);
		multiply();
		push(Y);			// 2nd root
		push(B);
		add();
		negate();
		push(A);
		divide();
		push_rational(1, 2);
		multiply();
		restore();
		return;
	}
	tos -= n;
	restore();
}

// use these for tmp vars (required by garbage collector)

U *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7, *p8, *p9;

int expanding;
int esc_flag;
int draw_flag;
int trigmode;
int term_flag;
int running;
jmp_buf stop_return, draw_stop_return;

void *
run1(void *s)
{
	run((char *) s);
	running = 0;
	return NULL;
}

void
run(char *s)
{
	int i, n;
	if (setjmp(stop_return))
		return;
	term_flag = BLACK;
	tos = 0;
	esc_flag = 0;
	draw_flag = 0;
	frame = stack + TOS;
	p0 = symbol(NIL);
	p1 = symbol(NIL);
	p2 = symbol(NIL);
	p3 = symbol(NIL);
	p4 = symbol(NIL);
	p5 = symbol(NIL);
	p6 = symbol(NIL);
	p7 = symbol(NIL);
	p8 = symbol(NIL);
	p9 = symbol(NIL);
	set_binding(symbol(TRACE), zero); // start with trace disabled
	while (1) {
		n = scan(s);
		p1 = pop();
		check_stack();
		if (n == 0)
			break;
		// if debug mode then print the source text
		if (equaln(get_binding(symbol(TRACE)), 1)) {
			for (i = 0; i < n; i++)
				if (s[i] != '\r')
					printchar(s[i]);
			if (s[n - 1] != '\n') // n is not zero, see above
				printchar('\n');
		}
		s += n;
		trigmode = 0;
		if (equaln(get_binding(symbol(AUTOEXPAND)), 0))
			expanding = 0;
		else
			expanding = 1;
		push(p1);
		eval_and_print_result(1);
		check_stack();
	}
}

void
check_stack(void)
{
	if (tos != 0)
		stop("stack error");
	if (frame != stack + TOS)
		stop("frame error");
}

// cannot reference symbols yet

void
echo_input(char *s)
{
	term_flag = BLUE;
	printstr(s);
	printstr("\n");
	term_flag = BLACK;
}

void
check_esc_flag(void)
{
	if (esc_flag)
		stop(NULL);
}

void
stop(char *s)
{
	if (draw_flag == 2)
		longjmp(draw_stop_return, 1);
	else {
		term_flag = RED;
		if (s == NULL)
			printstr("Stop\n");
		else {
			printstr("Stop: ");
			printstr(s);
			printstr("\n");
		}
		term_flag = BLACK;
		longjmp(stop_return, 1);
	}
}

// This scanner uses the recursive descent method.
//
// The char pointers token_str and scan_str are pointers to the input string as
// in the following example.
//
//	| g | a | m | m | a |   | a | l | p | h | a |
//	  ^                   ^
//	  token_str           scan_str
//
// The char pointer token_buf points to a malloc buffer.
//
//	| g | a | m | m | a | \0 |
//	  ^
//	  token_buf

#define T_INTEGER 1001
#define T_DOUBLE 1002
#define T_SYMBOL 1003
#define T_FUNCTION 1004
#define T_NEWLINE 1006
#define T_STRING 1007
#define T_GTEQ 1008
#define T_LTEQ 1009
#define T_EQ 1010

int token, newline_flag, meta_mode;
char *input_str, *scan_str, *token_str, *token_buf;

// Returns number of chars scanned and expr on stack.

// Returns zero when nothing left to scan.

int
scan(char *s)
{
	meta_mode = 0;
	expanding++;
	input_str = s;
	scan_str = s;
	get_next_token();
	if (token == 0) {
		push(symbol(NIL));
		expanding--;
		return 0;
	}
	scan_stmt();
	expanding--;
	return (int) (token_str - input_str);
}

int
scan_meta(char *s)
{
	meta_mode = 1;
	expanding++;
	input_str = s;
	scan_str = s;
	get_next_token();
	if (token == 0) {
		push(symbol(NIL));
		expanding--;
		return 0;
	}
	scan_stmt();
	expanding--;
	return (int) (token_str - input_str);
}

void
scan_stmt(void)
{
	scan_relation();
	if (token == '=') {
		get_next_token();
		push_symbol(SETQ);
		swap();
		scan_relation();
		list(3);
	}
}

void
scan_relation(void)
{
	scan_expression();
	switch (token) {
	case T_EQ:
		push_symbol(TESTEQ);
		swap();
		get_next_token();
		scan_expression();
		list(3);
		break;
	case T_LTEQ:
		push_symbol(TESTLE);
		swap();
		get_next_token();
		scan_expression();
		list(3);
		break;
	case T_GTEQ:
		push_symbol(TESTGE);
		swap();
		get_next_token();
		scan_expression();
		list(3);
		break;
	case '<':
		push_symbol(TESTLT);
		swap();
		get_next_token();
		scan_expression();
		list(3);
		break;
	case '>':
		push_symbol(TESTGT);
		swap();
		get_next_token();
		scan_expression();
		list(3);
		break;
	default:
		break;
	}
}

void
scan_expression(void)
{
	int h = tos;
	switch (token) {
	case '+':
		get_next_token();
		scan_term();
		break;
	case '-':
		get_next_token();
		scan_term();
		negate();
		break;
	default:
		scan_term();
		break;
	}
	while (newline_flag == 0 && (token == '+' || token == '-')) {
		if (token == '+') {
			get_next_token();
			scan_term();
		} else {
			get_next_token();
			scan_term();
			negate();
		}
	}
	if (tos - h > 1) {
		list(tos - h);
		push_symbol(ADD);
		swap();
		cons();
	}
}

int
is_factor(void)
{
	switch (token) {
	case '*':
	case '/':
		return 1;
	case '(':
	case T_SYMBOL:
	case T_FUNCTION:
	case T_INTEGER:
	case T_DOUBLE:
	case T_STRING:
		if (newline_flag) {	// implicit mul can't cross line
			scan_str = token_str;	// better error display
			return 0;
		} else
			return 1;
	default:
		break;
	}
	return 0;
}

void
scan_term(void)
{
	int h = tos;
	scan_power();
	// discard integer 1
	if (tos > h && isrational(stack[tos - 1]) && equaln(stack[tos - 1], 1))
		pop();
	while (is_factor()) {
		if (token == '*') {
			get_next_token();
			scan_power();
		} else if (token == '/') {
			get_next_token();
			scan_power();
			inverse();
		} else
			scan_power();
		// fold constants
		if (tos > h + 1 && isnum(stack[tos - 2]) && isnum(stack[tos - 1]))
			multiply();
		// discard integer 1
		if (tos > h && isrational(stack[tos - 1]) && equaln(stack[tos - 1], 1))
			pop();
	}
	if (h == tos)
		push_integer(1);
	else if (tos - h > 1) {
		list(tos - h);
		push_symbol(MULTIPLY);
		swap();
		cons();
	}
}

void
scan_power(void)
{
	scan_factor();
	if (token == '^') {
		get_next_token();
		push_symbol(POWER);
		swap();
		scan_power();
		list(3);
	}
}

void
scan_factor(void)
{
	int h;
	h = tos;
	if (token == '(')
		scan_subexpr();
	else if (token == T_SYMBOL)
		scan_symbol();
	else if (token == T_FUNCTION)
		scan_function_call();
	else if (token == T_INTEGER) {
		bignum_scan_integer(token_buf);
		get_next_token();
	} else if (token == T_DOUBLE) {
		bignum_scan_float(token_buf);
		get_next_token();
	} else if (token == T_STRING)
		scan_string();
	else
		error("syntax error");
	// index
	if (token == '[') {
		get_next_token();
		push_symbol(INDEX);
		swap();
		scan_expression();
		while (token == ',') {
			get_next_token();
			scan_expression();
		}
		if (token != ']')
			error("] expected");
		get_next_token();
		list(tos - h);
	}
	while (token == '!') {
		get_next_token();
		push_symbol(FACTORIAL);
		swap();
		list(2);
	}
}

void
scan_symbol(void)
{
	if (token != T_SYMBOL)
		error("symbol expected");
	if (meta_mode && strlen(token_buf) == 1)
		switch (token_buf[0]) {
		case 'a':
			push(symbol(METAA));
			break;
		case 'b':
			push(symbol(METAB));
			break;
		case 'x':
			push(symbol(METAX));
			break;
		default:
			push(usr_symbol(token_buf));
			break;
		}
	else
		push(usr_symbol(token_buf));
	get_next_token();
}

void
scan_string(void)
{
	new_string(token_buf);
	get_next_token();
}

void
scan_function_call(void)
{
	int n = 1;
	U *p;
	p = usr_symbol(token_buf);
	push(p);
	get_next_token();	// function name
	get_next_token();	// left paren
	if (token != ')') {
		scan_stmt();
		n++;
		while (token == ',') {
			get_next_token();
			scan_stmt();
			n++;
		}
	}
	if (token != ')')
		error(") expected");
	get_next_token();
	list(n);
}

// scan subexpression

void
scan_subexpr(void)
{
	int n;
	if (token != '(')
		error("( expected");
	get_next_token();
	scan_stmt();
	if (token == ',') {
		n = 1;
		while (token == ',') {
			get_next_token();
			scan_stmt();
			n++;
		}
		build_tensor(n);
	}
	if (token != ')')
		error(") expected");
	get_next_token();
}

void
error(char *errmsg)
{
	printchar('\n');
	// try not to put question mark on orphan line
	while (input_str != scan_str) {
		if ((*input_str == '\n' || *input_str == '\r') && input_str + 1 == scan_str)
			break;
		printchar(*input_str++);
	}
	printstr(" ??? ");
	while (*input_str && (*input_str != '\n' && *input_str != '\r'))
		printchar(*input_str++);
	printchar('\n');
	stop(errmsg);
}

// There are n expressions on the stack, possibly tensors.
//
// This function assembles the stack expressions into a single tensor.
//
// For example, at the top level of the expression ((a,b),(c,d)), the vectors
// (a,b) and (c,d) would be on the stack.

void
build_tensor(int n)
{
	// int i, j, k, ndim, nelem;
	int i;
	U **s;
	save();
	s = stack + tos - n;
	p2 = alloc_tensor(n);
	p2->u.tensor->ndim = 1;
	p2->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++)
		p2->u.tensor->elem[i] = s[i];
	tos -= n;
	push(p2);
	restore();
}

void
get_next_token(void)
{
	newline_flag = 0;
	while (1) {
		get_token();
		if (token != T_NEWLINE)
			break;
		newline_flag = 1;
	}
}

void
get_token(void)
{
	// skip spaces
	while (isspace(*scan_str)) {
		if (*scan_str == '\n' || *scan_str == '\r') {
			token = T_NEWLINE;
			scan_str++;
			return;
		}
		scan_str++;
	}
	token_str = scan_str;
	// end of string?
	if (*scan_str == 0) {
		token = 0;
		return;
	}
	// number?
	if (isdigit(*scan_str) || *scan_str == '.') {
		while (isdigit(*scan_str))
			scan_str++;
		if (*scan_str == '.') {
			scan_str++;
			while (isdigit(*scan_str))
				scan_str++;
			if (*scan_str == 'e' && (scan_str[1] == '+' || scan_str[1] == '-' || isdigit(scan_str[1]))) {
				scan_str += 2;
				while (isdigit(*scan_str))
					scan_str++;
			}
			token = T_DOUBLE;
		} else
			token = T_INTEGER;
		update_token_buf(token_str, scan_str);
		return;
	}
	// symbol?
	if (isalpha(*scan_str)) {
		while (isalnum(*scan_str))
			scan_str++;
		if (*scan_str == '(')
			token = T_FUNCTION;
		else
			token = T_SYMBOL;
		update_token_buf(token_str, scan_str);
		return;
	}
	// string ?
	if (*scan_str == '"') {
		scan_str++;
		while (*scan_str != '"') {
			if (*scan_str == 0 || *scan_str == '\n' || *scan_str == '\r')
				error("runaway string");
			scan_str++;
		}
		scan_str++;
		token = T_STRING;
		update_token_buf(token_str + 1, scan_str - 1);
		return;
	}
	// comment?
	if (*scan_str == '#' || (*scan_str == '-' && scan_str[1] == '-')) {
		while (*scan_str && *scan_str != '\n' && *scan_str != '\r')
			scan_str++;
		if (*scan_str)
			scan_str++;
		token = T_NEWLINE;
		return;
	}
	// relational operator?
	if (*scan_str == '=' && scan_str[1] == '=') {
		scan_str += 2;
		token = T_EQ;
		return;
	}
	if (*scan_str == '<' && scan_str[1] == '=') {
		scan_str += 2;
		token = T_LTEQ;
		return;
	}
	if (*scan_str == '>' && scan_str[1] == '=') {
		scan_str += 2;
		token = T_GTEQ;
		return;
	}
	// single char token
	token = *scan_str++;
}

void
update_token_buf(char *a, char *b)
{
	int n;
	if (token_buf)
		free(token_buf);
	n = (int) (b - a);
	token_buf = (char *) malloc(n + 1);
	if (token_buf == 0)
		stop("malloc failure");
	strncpy(token_buf, a, n);
	token_buf[n] = 0;
}

//	Notes:
//
//	Formerly add() and multiply() were used to construct expressions but
//	this preevaluation caused problems.
//
//	For example, suppose A has the floating point value inf.
//
//	Before, the expression A/A resulted in 1 because the scanner would
//	divide the symbols.
//
//	After removing add() and multiply(), A/A results in nan which is the
//	correct result.
//
//	The functions negate() and inverse() are used but they do not cause
//	problems with preevaluation of symbols.

void
eval_sgn(void)
{
	push(cadr(p1));
	eval();
	sgn();
}

void
sgn(void)
{
	save();
	p1 = pop();
	if (!isnum(p1)) {
		push_symbol(SGN);
		push(p1);
		list(2);
	} else if (iszero(p1))
		push_integer(0);
	else if (isnegativenumber(p1))
		push_integer(-1);
	else
		push_integer(1);
	restore();
}

// Simplify factorials
//
// The following script
//
//	F(n,k) = k binomial(n,k)
//	(F(n,k) + F(n,k-1)) / F(n+1,k)
//
// generates
//
//        k! n!             n! (1 - k + n)!              k! n!
// -------------------- + -------------------- - ----------------------
//  (-1 + k)! (1 + n)!     (1 + n)! (-k + n)!     k (-1 + k)! (1 + n)!
//
// Simplify each term to get
//
//    k       1 - k + n       1
// ------- + ----------- - -------
//  1 + n       1 + n       1 + n
//
// Then simplify the sum to get
//
//    n
// -------
//  1 + n

// simplify factorials term-by-term

void
eval_simfac(void)
{
	push(cadr(p1));
	eval();
	simfac();
}

#if 1

void
simfac(void)
{
	int h;
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD)) {
		h = tos;
		p1 = cdr(p1);
		while (p1 != symbol(NIL)) {
			push(car(p1));
			simfac_term();
			p1 = cdr(p1);
		}
		add_all(tos - h);
	} else {
		push(p1);
		simfac_term();
	}
	restore();
}

#else

void
simfac(void)
{
	int h;
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD)) {
		h = tos;
		p1 = cdr(p1);
		while (p1 != symbol(NIL)) {
			push(car(p1));
			simfac_term();
			p1 = cdr(p1);
		}
		addk(tos - h);
		p1 = pop();
		if (find(p1, symbol(FACTORIAL))) {
			push(p1);
			if (car(p1) == symbol(ADD)) {
				Condense();
				simfac_term();
			}
		}
	} else {
		push(p1);
		simfac_term();
	}
	restore();
}

#endif

void
simfac_term(void)
{
	int h;
	save();
	p1 = pop();
	// if not a product of factors then done
	if (car(p1) != symbol(MULTIPLY)) {
		push(p1);
		restore();
		return;
	}
	// push all factors
	h = tos;
	p1 = cdr(p1);
	while (p1 != symbol(NIL)) {
		push(car(p1));
		p1 = cdr(p1);
	}
	// keep trying until no more to do
	while (yysimfac(h))
		;
	multiply_all_noexpand(tos - h);
	restore();
}

// try all pairs of factors

int
yysimfac(int h)
{
	int i, j;
	for (i = h; i < tos; i++) {
		p1 = stack[i];
		for (j = h; j < tos; j++) {
			if (i == j)
				continue;
			p2 = stack[j];
			//	n! / n		->	(n - 1)!
			if (car(p1) == symbol(FACTORIAL)
			&& car(p2) == symbol(POWER)
			&& isminusone(caddr(p2))
			&& equal(cadr(p1), cadr(p2))) {
				push(cadr(p1));
				push(one);
				subtract();
				factorial();
				stack[i] = pop();
				stack[j] = one;
				return 1;
			}
			//	n / n!		->	1 / (n - 1)!
			if (car(p2) == symbol(POWER)
			&& isminusone(caddr(p2))
			&& caadr(p2) == symbol(FACTORIAL)
			&& equal(p1, cadadr(p2))) {
				push(p1);
				push_integer(-1);
				add();
				factorial();
				reciprocate();
				stack[i] = pop();
				stack[j] = one;
				return 1;
			}
			//	(n + 1) n!	->	(n + 1)!
			if (car(p2) == symbol(FACTORIAL)) {
				push(p1);
				push(cadr(p2));
				subtract();
				p3 = pop();
				if (isplusone(p3)) {
					push(p1);
					factorial();
					stack[i] = pop();
					stack[j] = one;
					return 1;
				}
			}
			//	1 / ((n + 1) n!)	->	1 / (n + 1)!
			if (car(p1) == symbol(POWER)
			&& isminusone(caddr(p1))
			&& car(p2) == symbol(POWER)
			&& isminusone(caddr(p2))
			&& caadr(p2) == symbol(FACTORIAL)) {
				push(cadr(p1));
				push(cadr(cadr(p2)));
				subtract();
				p3 = pop();
				if (isplusone(p3)) {
					push(cadr(p1));
					factorial();
					reciprocate();
					stack[i] = pop();
					stack[j] = one;
					return 1;
				}
			}
			//	(n + 1)! / n!	->	n + 1
			//	n! / (n + 1)!	->	1 / (n + 1)
			if (car(p1) == symbol(FACTORIAL)
			&& car(p2) == symbol(POWER)
			&& isminusone(caddr(p2))
			&& caadr(p2) == symbol(FACTORIAL)) {
				push(cadr(p1));
				push(cadr(cadr(p2)));
				subtract();
				p3 = pop();
				if (isplusone(p3)) {
					stack[i] = cadr(p1);
					stack[j] = one;
					return 1;
				}
				if (isminusone(p3)) {
					push(cadr(cadr(p2)));
					reciprocate();
					stack[i] = pop();
					stack[j] = one;
					return 1;
				}
				if (equaln(p3, 2)) {
					stack[i] = cadr(p1);
					push(cadr(p1));
					push_integer(-1);
					add();
					stack[j] = pop();
					return 1;
				}
				if (equaln(p3, -2)) {
					push(cadr(cadr(p2)));
					reciprocate();
					stack[i] = pop();
					push(cadr(cadr(p2)));
					push_integer(-1);
					add();
					reciprocate();
					stack[j] = pop();
					return 1;
				}
			}
		}
	}
	return 0;
}

extern int trigmode;

void
eval_simplify(void)
{
	push(cadr(p1));
	eval();
	simplify();
}

void
simplify(void)
{
	save();
	simplify_main();
	restore();
}

void
simplify_main(void)
{
	p1 = pop();
	if (istensor(p1)) {
		simplify_tensor();
		return;
	}
	if (find(p1, symbol(FACTORIAL))) {
		push(p1);
		simfac();
		p2 = pop();
		push(p1);
		rationalize();
		simfac();
		p3 = pop();
		if (count(p2) < count(p3))
			p1 = p2;
		else
			p1 = p3;
	}
	f1();
	f2();
	f3();
	f4();
	f5();
	f9();
	push(p1);
}

void
simplify_tensor(void)
{
	int i;
	p2 = alloc_tensor(p1->u.tensor->nelem);
	p2->u.tensor->ndim = p1->u.tensor->ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		p2->u.tensor->dim[i] = p1->u.tensor->dim[i];
	for (i = 0; i < p1->u.tensor->nelem; i++) {
		push(p1->u.tensor->elem[i]);
		simplify();
		p2->u.tensor->elem[i] = pop();
	}
	if (iszero(p2))
		p2 = zero; // null tensor becomes scalar zero
	push(p2);
}

int
count(U *p)
{
	int n;
	if (iscons(p)) {
		n = 0;
		while (iscons(p)) {
			n += count(car(p)) + 1;
			p = cdr(p);
		}
	} else
		n = 1;
	return n;
}

// try rationalizing

void
f1(void)
{
	if (car(p1) != symbol(ADD))
		return;
	push(p1);
	rationalize();
	p2 = pop();
	if (count(p2) < count(p1))
		p1 = p2;
}

// try condensing

void
f2(void)
{
	if (car(p1) != symbol(ADD))
		return;
	push(p1);
	Condense();
	p2 = pop();
	if (count(p2) <= count(p1))
		p1 = p2;
}

// this simplifies forms like (A-B) / (B-A)

void
f3(void)
{
	push(p1);
	rationalize();
	negate();
	rationalize();
	negate();
	rationalize();
	p2 = pop();
	if (count(p2) < count(p1))
		p1 = p2;
}

// try expanding denominators

void
f4(void)
{
	if (iszero(p1))
		return;
	push(p1);
	rationalize();
	inverse();
	rationalize();
	inverse();
	rationalize();
	p2 = pop();
	if (count(p2) < count(p1))
		p1 = p2;
}

// simplifies trig forms

void
simplify_trig(void)
{
	save();
	p1 = pop();
	f5();
	push(p1);
	restore();
}

void
f5(void)
{
	if (find(p1, symbol(SIN)) == 0 && find(p1, symbol(COS)) == 0)
		return;
	p2 = p1;
	trigmode = 1;
	push(p2);
	eval();
	p3 = pop();
	trigmode = 2;
	push(p2);
	eval();
	p4 = pop();
	trigmode = 0;
	if (count(p4) < count(p3) || nterms(p4) < nterms(p3))
		p3 = p4;
	if (count(p3) < count(p1) || nterms(p3) < nterms(p1))
		p1 = p3;
}

// if it's a sum then try to simplify each term

void
f9(void)
{
	if (car(p1) != symbol(ADD))
		return;
	push_integer(0);
	p2 = cdr(p1);
	while (iscons(p2)) {
		push(car(p2));
		simplify();
		add();
		p2 = cdr(p2);
	}
	p2 = pop();
	if (count(p2) < count(p1))
		p1 = p2;
}

int
nterms(U *p)
{
	if (car(p) != symbol(ADD))
		return 1;
	else
		return length(p) - 1;
}

void
eval_sin(void)
{
	push(cadr(p1));
	eval();
	sine();
}

void
sine(void)
{
	save();
	p1 = pop();
	if (car(p1) == symbol(ADD))
		sine_of_angle_sum();
	else
		sine_of_angle();
	restore();
}

// Use angle sum formula for special angles.

#undef A
#undef B

#define A p3
#define B p4

void
sine_of_angle_sum(void)
{
	p2 = cdr(p1);
	while (iscons(p2)) {
		B = car(p2);
		if (isnpi(B)) {
			push(p1);
			push(B);
			subtract();
			A = pop();
			push(A);
			sine();
			push(B);
			cosine();
			multiply();
			push(A);
			cosine();
			push(B);
			sine();
			multiply();
			add();
			return;
		}
		p2 = cdr(p2);
	}
	sine_of_angle();
}

void
sine_of_angle(void)
{
	int n;
	double d;
	if (car(p1) == symbol(ARCSIN)) {
		push(cadr(p1));
		return;
	}
	if (isdouble(p1)) {
		d = sin(p1->u.d);
		if (fabs(d) < 1e-10)
			d = 0.0;
		push_double(d);
		return;
	}
	// sine function is antisymmetric, sin(-x) = -sin(x)
	if (isnegative(p1)) {
		push(p1);
		negate();
		sine();
		negate();
		return;
	}
	// sin(arctan(x)) = x / sqrt(1 + x^2)
	// see p. 173 of the CRC Handbook of Mathematical Sciences
	if (car(p1) == symbol(ARCTAN)) {
		push(cadr(p1));
		push_integer(1);
		push(cadr(p1));
		push_integer(2);
		power();
		add();
		push_rational(-1, 2);
		power();
		multiply();
		return;
	}
	// multiply by 180/pi
	push(p1);
	push_integer(180);
	multiply();
	push_symbol(PI);
	divide();
	n = pop_integer();
	if (n < 0) {
		push(symbol(SIN));
		push(p1);
		list(2);
		return;
	}
	switch (n % 360) {
	case 0:
	case 180:
		push_integer(0);
		break;
	case 30:
	case 150:
		push_rational(1, 2);
		break;
	case 210:
	case 330:
		push_rational(-1, 2);
		break;
	case 45:
	case 135:
		push_rational(1, 2);
		push_integer(2);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 225:
	case 315:
		push_rational(-1, 2);
		push_integer(2);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 60:
	case 120:
		push_rational(1, 2);
		push_integer(3);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 240:
	case 300:
		push_rational(-1, 2);
		push_integer(3);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 90:
		push_integer(1);
		break;
	case 270:
		push_integer(-1);
		break;
	default:
		push(symbol(SIN));
		push(p1);
		list(2);
		break;
	}
}

//	          exp(x) - exp(-x)
//	sinh(x) = ----------------
//	                 2

void
eval_sinh(void)
{
	push(cadr(p1));
	eval();
	ysinh();
}

void
ysinh(void)
{
	save();
	yysinh();
	restore();
}

void
yysinh(void)
{
	double d;
	p1 = pop();
	if (car(p1) == symbol(ARCSINH)) {
		push(cadr(p1));
		return;
	}
	if (isdouble(p1)) {
		d = sinh(p1->u.d);
		if (fabs(d) < 1e-10)
			d = 0.0;
		push_double(d);
		return;
	}
	if (iszero(p1)) {
		push(zero);
		return;
	}
	push_symbol(SINH);
	push(p1);
	list(2);
}

//	 _______
//	|	| <- stack
//	|	|
//	|_______|
//	|	| <- stack + tos
//	|	|
//	|	|
//	|_______|
//	|	| <- frame
//	|_______|
//		  <- stack + TOS
//
//	The stack grows from low memory towards high memory. This is so that
//	multiple expressions can be pushed on the stack and then accessed as an
//	array.
//
//	The frame area holds local variables and grows from high memory towards
//	low memory. The frame area makes local variables visible to the garbage
//	collector.

U **frame, *stack[TOS];
int tos;

void
push(U *p)
{
	if (stack + tos >= frame)
		stop("memory full");
	stack[tos++] = p;
}

U *
pop(void)
{
	if (tos == 0)
		stop("stack error");
	return stack[--tos];
}

void
push_frame(int n)
{
	int i;
	frame -= n;
	if (frame < stack + tos)
		stop("memory full");
	for (i = 0; i < n; i++)
		frame[i] = symbol(NIL);
}

void
pop_frame(int n)
{
	frame += n;
	if (frame > stack + TOS)
		stop("frame error");
}

void
save(void)
{
	frame -= 10;
	if (frame < stack + tos)
		stop("memory full");
	frame[0] = p0;
	frame[1] = p1;
	frame[2] = p2;
	frame[3] = p3;
	frame[4] = p4;
	frame[5] = p5;
	frame[6] = p6;
	frame[7] = p7;
	frame[8] = p8;
	frame[9] = p9;
}

void
restore(void)
{
	if (frame > stack + TOS - 10)
		stop("frame error");
	p0 = frame[0];
	p1 = frame[1];
	p2 = frame[2];
	p3 = frame[3];
	p4 = frame[4];
	p5 = frame[5];
	p6 = frame[6];
	p7 = frame[7];
	p8 = frame[8];
	p9 = frame[9];
	frame += 10;
}

// Local U * is OK here because there is no functional path to the garbage collector.

void
swap(void)
{
	U *p, *q;
	p = pop();
	q = pop();
	push(p);
	push(q);
}

// Local U * is OK here because there is no functional path to the garbage collector.

void
dupl(void)
{
	U *p;
	p = pop();
	push(p);
	push(p);
}

/*	Substitute new expr for old expr in expr.

	Input:	push	expr

		push	old expr

		push	new expr

	Output:	Result on stack
*/

void
subst(void)
{
	int i;
	save();
	p3 = pop(); // new expr
	p2 = pop(); // old expr
	if (p2 == symbol(NIL) || p3 == symbol(NIL)) {
		restore();
		return;
	}
	p1 = pop(); // expr
	if (istensor(p1)) {
		p4 = alloc_tensor(p1->u.tensor->nelem);
		p4->u.tensor->ndim = p1->u.tensor->ndim;
		for (i = 0; i < p1->u.tensor->ndim; i++)
			p4->u.tensor->dim[i] = p1->u.tensor->dim[i];
		for (i = 0; i < p1->u.tensor->nelem; i++) {
			push(p1->u.tensor->elem[i]);
			push(p2);
			push(p3);
			subst();
			p4->u.tensor->elem[i] = pop();
		}
		push(p4);
	} else if (equal(p1, p2))
		push(p3);
	else if (iscons(p1)) {
		push(car(p1));
		push(p2);
		push(p3);
		subst();
		push(cdr(p1));
		push(p2);
		push(p3);
		subst();
		cons();
	} else
		push(p1);
	restore();
}

// 'sum' function

#undef I
#undef X

#define I p5
#define X p6

void
eval_sum(void)
{
	int i, j, k;
	// 1st arg (quoted)
	X = cadr(p1);
	if (!issymbol(X))
		stop("sum: 1st arg?");
	// 2nd arg
	push(caddr(p1));
	eval();
	j = pop_integer();
	if (j == (int) 0x80000000)
		stop("sum: 2nd arg?");
	// 3rd arg
	push(cadddr(p1));
	eval();
	k = pop_integer();
	if (k == (int) 0x80000000)
		stop("sum: 3rd arg?");
	// 4th arg
	p1 = caddddr(p1);
	push_binding(X);
	push_integer(0);
	for (i = j; i <= k; i++) {
		push_integer(i);
		set_binding(X, pop());
		push(p1);
		eval();
		add();
	}
	p1 = pop();
	pop_binding(X);
	push(p1);
}

// The symbol table is a simple array of struct U.

U symtab[NSYM];
U *binding[NSYM];
U *arglist[NSYM];

// put symbol at index n

void
std_symbol(char *s, int n)
{
	U *p;
	p = symtab + n;
	p->u.printname = strdup(s);
}

// symbol lookup, create symbol if need be

U *
usr_symbol(char *s)
{
	int i;
	U *p;
	for (i = 0; i < NSYM; i++) {
		if (symtab[i].u.printname == 0)
			break;
		if (strcmp(s, symtab[i].u.printname) == 0)
			return symtab + i;
	}
	if (i == NSYM)
		stop("symbol table overflow");
	p = symtab + i;
	p->u.printname = strdup(s);
	return p;
}

// get the symbol's printname

char *
get_printname(U *p)
{
	if (p->k != SYM)
		stop("symbol error");
	return p->u.printname;
}

void
set_binding(U *p, U *b)
{
	if (p->k != SYM || p - symtab < MARK2) {
		printf("%d %d\n", MARK2, (int) (p - symtab));
		stop("reserved symbol");
	}
	binding[p - symtab] = b;
	arglist[p - symtab] = symbol(NIL);
}

void
set_binding_and_arglist(U *p, U *b, U *a)
{
	if (p->k != SYM || p - symtab < MARK2)
		stop("reserved symbol");
	binding[p - symtab] = b;
	arglist[p - symtab] = a;
}

U *
get_binding(U *p)
{
	if (p->k != SYM)
		stop("symbol error");
	return binding[p - symtab];
}

U *
get_arglist(U *p)
{
	if (p->k != SYM)
		stop("symbol error");
	return arglist[p - symtab];
}

// get symbol's number from ptr

int
symnum(U *p)
{
	if (p->k != SYM)
		stop("symbol error");
	return (int) (p - symtab);
}

void
push_binding(U *p)
{
	if (p->k != SYM)
		stop("symbol expected");
	push(binding[p - symtab]);
	push(arglist[p - symtab]);
}

void
pop_binding(U *p)
{
	if (p->k != SYM)
		stop("symbol expected");
	arglist[p - symtab] = pop();
	binding[p - symtab] = pop();
}

void
eval_tan(void)
{
	push(cadr(p1));
	eval();
	tangent();
}

void
tangent(void)
{
	save();
	yytangent();
	restore();
}

void
yytangent(void)
{
	int n;
	double d;
	p1 = pop();
	if (car(p1) == symbol(ARCTAN)) {
		push(cadr(p1));
		return;
	}
	if (isdouble(p1)) {
		d = tan(p1->u.d);
		if (fabs(d) < 1e-10)
			d = 0.0;
		push_double(d);
		return;
	}
	// tan function is antisymmetric, tan(-x) = -tan(x)
	if (isnegative(p1)) {
		push(p1);
		negate();
		tangent();
		negate();
		return;
	}
	// multiply by 180/pi
	push(p1);
	push_integer(180);
	multiply();
	push_symbol(PI);
	divide();
	n = pop_integer();
	if (n < 0) {
		push_symbol(TAN);
		push(p1);
		list(2);
		return;
	}
	switch (n % 360) {
	case 0:
	case 180:
		push_integer(0);
		break;
	case 30:
	case 210:
		push_rational(1, 3);
		push_integer(3);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 150:
	case 330:
		push_rational(-1, 3);
		push_integer(3);
		push_rational(1, 2);
		power();
		multiply();
		break;
	case 45:
	case 225:
		push_integer(1);
		break;
	case 135:
	case 315:
		push_integer(-1);
		break;
	case 60:
	case 240:
		push_integer(3);
		push_rational(1, 2);
		power();
		break;
	case 120:
	case 300:
		push_integer(3);
		push_rational(1, 2);
		power();
		negate();
		break;
	default:
		push(symbol(TAN));
		push(p1);
		list(2);
		break;
	}
}

//	           exp(2 x) - 1
//	tanh(x) = --------------
//	           exp(2 x) + 1

void
eval_tanh(void)
{
	double d;
	push(cadr(p1));
	eval();
	p1 = pop();
	if (car(p1) == symbol(ARCTANH)) {
		push(cadr(p1));
		return;
	}
	if (isdouble(p1)) {
		d = tanh(p1->u.d);
		if (fabs(d) < 1e-10)
			d = 0.0;
		push_double(d);
		return;
	}
	if (iszero(p1)) {
		push(zero);
		return;
	}
	push(symbol(TANH));
	push(p1);
	list(2);
}

/* Taylor expansion of a function

	push(F)
	push(X)
	push(N)
	push(A)
	taylor()
*/

void
eval_taylor(void)
{
	// 1st arg
	p1 = cdr(p1);
	push(car(p1));
	eval();
	// 2nd arg
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		guess();
	else
		push(p2);
	// 3rd arg
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		push_integer(24); // default number of terms
	else
		push(p2);
	// 4th arg
	p1 = cdr(p1);
	push(car(p1));
	eval();
	p2 = pop();
	if (p2 == symbol(NIL))
		push_integer(0); // default expansion point
	else
		push(p2);
	taylor();
}

#undef F
#undef X
#undef N
#undef A
#undef C

#define F p1
#define X p2
#define N p3
#define A p4
#define C p5

void
taylor(void)
{
	int i, k;
	save();
	A = pop();
	N = pop();
	X = pop();
	F = pop();
	push(N);
	k = pop_integer();
	if (k == (int) 0x80000000) {
		push(symbol(TAYLOR));
		push(F);
		push(X);
		push(N);
		push(A);
		list(5);
		restore();
		return;
	}
	push(F);	// f(a)
	push(X);
	push(A);
	subst();
	eval();
	push_integer(1);
	C = pop();
	for (i = 1; i <= k; i++) {
		push(F);	// f = f'
		push(X);
		derivative();
		F = pop();
		if (iszero(F))
			break;
		push(C);	// c = c * (x - a)
		push(X);
		push(A);
		subtract();
		multiply();
		C = pop();
		push(F);	// f(a)
		push(X);
		push(A);
		subst();
		eval();
		push(C);
		multiply();
		push_integer(i);
		factorial();
		divide();
		add();
	}
	restore();
}

//-----------------------------------------------------------------------------
//
//	Called from the "eval" module to evaluate tensor elements.
//
//	p1 points to the tensor operand.
//
//-----------------------------------------------------------------------------

void
eval_tensor(void)
{
	int i, ndim, nelem;
	U **a, **b;
	//---------------------------------------------------------------------
	//
	//	create a new tensor for the result
	//
	//---------------------------------------------------------------------
	nelem = p1->u.tensor->nelem;
	ndim = p1->u.tensor->ndim;
	p2 = alloc_tensor(nelem);
	p2->u.tensor->ndim = ndim;
	for (i = 0; i < ndim; i++)
		p2->u.tensor->dim[i] = p1->u.tensor->dim[i];
	//---------------------------------------------------------------------
	//
	//	b = eval(a)
	//
	//---------------------------------------------------------------------
	a = p1->u.tensor->elem;
	b = p2->u.tensor->elem;
	for (i = 0; i < nelem; i++) {
		push(a[i]);
		eval();
		b[i] = pop();
	}
	//---------------------------------------------------------------------
	//
	//	push the result
	//
	//---------------------------------------------------------------------
	push(p2);
	promote_tensor();
}

//-----------------------------------------------------------------------------
//
//	Add tensors
//
//	Input:		Operands on stack
//
//	Output:		Result on stack
//
//-----------------------------------------------------------------------------

void
tensor_plus_tensor(void)
{
	int i, ndim, nelem;
	U **a, **b, **c;
	save();
	p2 = pop();
	p1 = pop();
	// are the dimension lists equal?
	ndim = p1->u.tensor->ndim;
	if (ndim != p2->u.tensor->ndim) {
		push(symbol(NIL));
		restore();
		return;
	}
	for (i = 0; i < ndim; i++)
		if (p1->u.tensor->dim[i] != p2->u.tensor->dim[i]) {
			push(symbol(NIL));
			restore();
			return;
		}
	// create a new tensor for the result
	nelem = p1->u.tensor->nelem;
	p3 = alloc_tensor(nelem);
	p3->u.tensor->ndim = ndim;
	for (i = 0; i < ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	// c = a + b
	a = p1->u.tensor->elem;
	b = p2->u.tensor->elem;
	c = p3->u.tensor->elem;
	for (i = 0; i < nelem; i++) {
		push(a[i]);
		push(b[i]);
		add();
		c[i] = pop();
	}
	// push the result
	push(p3);
	restore();
}

//-----------------------------------------------------------------------------
//
//	careful not to reorder factors
//
//-----------------------------------------------------------------------------

void
tensor_times_scalar(void)
{
	int i, ndim, nelem;
	U **a, **b;
	save();
	p2 = pop();
	p1 = pop();
	ndim = p1->u.tensor->ndim;
	nelem = p1->u.tensor->nelem;
	p3 = alloc_tensor(nelem);
	p3->u.tensor->ndim = ndim;
	for (i = 0; i < ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	a = p1->u.tensor->elem;
	b = p3->u.tensor->elem;
	for (i = 0; i < nelem; i++) {
		push(a[i]);
		push(p2);
		multiply();
		b[i] = pop();
	}
	push(p3);
	restore();
}

void
scalar_times_tensor(void)
{
	int i, ndim, nelem;
	U **a, **b;
	save();
	p2 = pop();
	p1 = pop();
	ndim = p2->u.tensor->ndim;
	nelem = p2->u.tensor->nelem;
	p3 = alloc_tensor(nelem);
	p3->u.tensor->ndim = ndim;
	for (i = 0; i < ndim; i++)
		p3->u.tensor->dim[i] = p2->u.tensor->dim[i];
	a = p2->u.tensor->elem;
	b = p3->u.tensor->elem;
	for (i = 0; i < nelem; i++) {
		push(p1);
		push(a[i]);
		multiply();
		b[i] = pop();
	}
	push(p3);
	restore();
}

int
is_square_matrix(U *p)
{
	if (istensor(p) && p->u.tensor->ndim == 2 && p->u.tensor->dim[0] == p->u.tensor->dim[1])
		return 1;
	else
		return 0;
}

//-----------------------------------------------------------------------------
//
//	gradient of tensor
//
//-----------------------------------------------------------------------------

void
d_tensor_tensor(void)
{
	int i, j, ndim, nelem;
	U **a, **b, **c;
	ndim = p1->u.tensor->ndim;
	nelem = p1->u.tensor->nelem;
	if (ndim + 1 >= MAXDIM)
		goto dont_evaluate;
	p3 = alloc_tensor(nelem * p2->u.tensor->nelem);
	p3->u.tensor->ndim = ndim + 1;
	for (i = 0; i < ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	p3->u.tensor->dim[ndim] = p2->u.tensor->dim[0];
	a = p1->u.tensor->elem;
	b = p2->u.tensor->elem;
	c = p3->u.tensor->elem;
	for (i = 0; i < nelem; i++) {
		for (j = 0; j < p2->u.tensor->nelem; j++) {
			push(a[i]);
			push(b[j]);
			derivative();
			c[i * p2->u.tensor->nelem + j] = pop();
		}
	}
	push(p3);
	return;
dont_evaluate:
	push(symbol(DERIVATIVE));
	push(p1);
	push(p2);
	list(3);
}

/* Generalized gradient of scalar

	p1	scalar expression

	p2	tensor
*/

void
d_scalar_tensor(void)
{
	int i, n;
	U **a, **b;
	push(p2);
	copy_tensor();
	p3 = pop();
	a = p2->u.tensor->elem;
	b = p3->u.tensor->elem;
	n = p2->u.tensor->nelem;
	for (i = 0; i < n; i++) {
		push(p1);
		push(a[i]);
		derivative();
		b[i] = pop();
	}
	push(p3);
}

//-----------------------------------------------------------------------------
//
//	Derivative of tensor
//
//-----------------------------------------------------------------------------

void
d_tensor_scalar(void)
{
	int i;
	U **a, **b;
	p3 = alloc_tensor(p1->u.tensor->nelem);
	p3->u.tensor->ndim = p1->u.tensor->ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	a = p1->u.tensor->elem;
	b = p3->u.tensor->elem;
	for (i = 0; i < p1->u.tensor->nelem; i++) {
		push(a[i]);
		push(p2);
		derivative();
		b[i] = pop();
	}
	push(p3);
}

int
compare_tensors(U *p1, U *p2)
{
	int i;
	if (p1->u.tensor->ndim < p2->u.tensor->ndim)
		return -1;
	if (p1->u.tensor->ndim > p2->u.tensor->ndim)
		return 1;
	for (i = 0; i < p1->u.tensor->ndim; i++) {
		if (p1->u.tensor->dim[i] < p2->u.tensor->dim[i])
			return -1;
		if (p1->u.tensor->dim[i] > p2->u.tensor->dim[i])
			return 1;
	}
	for (i = 0; i < p1->u.tensor->nelem; i++) {
		if (equal(p1->u.tensor->elem[i], p2->u.tensor->elem[i]))
			continue;
		if (lessp(p1->u.tensor->elem[i], p2->u.tensor->elem[i]))
			return -1;
		else
			return 1;
	}
	return 0;
}

//-----------------------------------------------------------------------------
//
//	Raise a tensor to a power
//
//	Input:		p1	tensor
//
//			p2	exponent
//
//	Output:		Result on stack
//
//-----------------------------------------------------------------------------

void
power_tensor(void)
{
	int i, k, n;
	// first and last dims must be equal
	k = p1->u.tensor->ndim - 1;
	if (p1->u.tensor->dim[0] != p1->u.tensor->dim[k]) {
		push(symbol(POWER));
		push(p1);
		push(p2);
		list(3);
		return;
	}
	push(p2);
	n = pop_integer();
	if (n == (int) 0x80000000) {
		push(symbol(POWER));
		push(p1);
		push(p2);
		list(3);
		return;
	}
	if (n == 0) {
		if (p1->u.tensor->ndim != 2)
			stop("power(tensor,0) with tensor rank not equal to 2");
		n = p1->u.tensor->dim[0];
		p1 = alloc_tensor(n * n);
		p1->u.tensor->ndim = 2;
		p1->u.tensor->dim[0] = n;
		p1->u.tensor->dim[1] = n;
		for (i = 0; i < n; i++)
			p1->u.tensor->elem[n * i + i] = one;
		push(p1);
		return;
	}
	if (n < 0) {
		n = -n;
		push(p1);
		inv();
		p1 = pop();
	}
	push(p1);
	for (i = 1; i < n; i++) {
		push(p1);
		inner();
		if (iszero(stack[tos - 1]))
			break;
	}
}

void
copy_tensor(void)
{
	int i;
	save();
	p1 = pop();
	p2 = alloc_tensor(p1->u.tensor->nelem);
	p2->u.tensor->ndim = p1->u.tensor->ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		p2->u.tensor->dim[i] = p1->u.tensor->dim[i];
	for (i = 0; i < p1->u.tensor->nelem; i++)
		p2->u.tensor->elem[i] = p1->u.tensor->elem[i];
	push(p2);
	restore();
}

// Tensors with elements that are also tensors get promoted to a higher rank.

void
promote_tensor(void)
{
	int i, j, k, nelem, ndim;
	save();
	p1 = pop();
	if (!istensor(p1)) {
		push(p1);
		restore();
		return;
	}
	p2 = p1->u.tensor->elem[0];
	for (i = 1; i < p1->u.tensor->nelem; i++)
		if (!compatible(p2, p1->u.tensor->elem[i]))
			stop("Cannot promote tensor due to inconsistent tensor components.");
	if (!istensor(p2)) {
		push(p1);
		restore();
		return;
	}
	ndim = p1->u.tensor->ndim + p2->u.tensor->ndim;
	if (ndim > MAXDIM)
		stop("tensor rank > 24");
	nelem = p1->u.tensor->nelem * p2->u.tensor->nelem;
	p3 = alloc_tensor(nelem);
	p3->u.tensor->ndim = ndim;
	for (i = 0; i < p1->u.tensor->ndim; i++)
		p3->u.tensor->dim[i] = p1->u.tensor->dim[i];
	for (j = 0; j < p2->u.tensor->ndim; j++)
		p3->u.tensor->dim[i + j] = p2->u.tensor->dim[j];
	k = 0;
	for (i = 0; i < p1->u.tensor->nelem; i++) {
		p2 = p1->u.tensor->elem[i];
		for (j = 0; j < p2->u.tensor->nelem; j++)
			p3->u.tensor->elem[k++] = p2->u.tensor->elem[j];
	}
	push(p3);
	restore();
}

int
compatible(U *p, U *q)
{
	int i;
	if (!istensor(p) && !istensor(q))
		return 1;
	if (!istensor(p) || !istensor(q))
		return 0;
	if (p->u.tensor->ndim != q->u.tensor->ndim)
		return 0;
	for (i = 0; i < p->u.tensor->ndim; i++)
		if (p->u.tensor->dim[i] != q->u.tensor->dim[i])
			return 0;
	return 1;
}

// If the number of args is odd then the last arg is the default result.

void
eval_test(void)
{
	p1 = cdr(p1);
	while (iscons(p1)) {
		if (cdr(p1) == symbol(NIL)) {
			push(car(p1)); // default case
			eval();
			return;
		}
		push(car(p1));
		eval_predicate();
		p2 = pop();
		if (!iszero(p2)) {
			push(cadr(p1));
			eval();
			return;
		}
		p1 = cddr(p1);
	}
	push_integer(0);
}

// The test for equality is weaker than the other relational operators.

// For example, A<=B causes a stop when the result of A minus B is not a
// numerical value.

// However, A==B never causes a stop.

// For A==B, any nonzero result for A minus B indicates inequality.

void
eval_testeq(void)
{
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	subtract();
	p1 = pop();
	if (iszero(p1))
		push_integer(1);
	else
		push_integer(0);
}

// Relational operators expect a numeric result for operand difference.

void
eval_testge(void)
{
	if (cmp_args() >= 0)
		push_integer(1);
	else
		push_integer(0);
}

void
eval_testgt(void)
{
	if (cmp_args() > 0)
		push_integer(1);
	else
		push_integer(0);
}

void
eval_testle(void)
{
	if (cmp_args() <= 0)
		push_integer(1);
	else
		push_integer(0);
}

void
eval_testlt(void)
{
	if (cmp_args() < 0)
		push_integer(1);
	else
		push_integer(0);
}

void
eval_not(void)
{
	push(cadr(p1));
	eval_predicate();
	p1 = pop();
	if (iszero(p1))
		push_integer(1);
	else
		push_integer(0);
}

void
eval_and(void)
{
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval_predicate();
		p2 = pop();
		if (iszero(p2)) {
			push_integer(0);
			return;
		}
		p1 = cdr(p1);
	}
	push_integer(1);
}

void
eval_or(void)
{
	p1 = cdr(p1);
	while (iscons(p1)) {
		push(car(p1));
		eval_predicate();
		p2 = pop();
		if (!iszero(p2)) {
			push_integer(1);
			return;
		}
		p1 = cdr(p1);
	}
	push_integer(0);
}

// use subtract for cases like A < A + 1

int
cmp_args(void)
{
	int t;
	push(cadr(p1));
	eval();
	push(caddr(p1));
	eval();
	subtract();
	p1 = pop();
	// try floating point if necessary
	if (p1->k != NUM && p1->k != DOUBLE) {
		push(p1);
		yyfloat();
		eval();
		p1 = pop();
	}
	if (iszero(p1))
		return 0;
	switch (p1->k) {
	case NUM:
		if (MSIGN(p1->u.q.a) == -1)
			t = -1;
		else
			t = 1;
		break;
	case DOUBLE:
		if (p1->u.d < 0.0)
			t = -1;
		else
			t = 1;
		break;
	default:
		stop("relational operator: cannot determine due to non-numerical comparison");
		t = 0;
	}
	return t;
}

/* Transform an expression using table look-up

The expression and free variable are on the stack.

The argument s is a null terminated list of transform rules.

For example, see itab.cpp

Internally, the following symbols are used:

	F	input expression

	X	free variable, i.e. F of X

	A	template expression

	B	result expression

	C	list of conditional expressions
*/

// p1 and p2 are tmps

#undef F
#undef X
#undef A
#undef B
#undef C

#define F p3
#define X p4
#define A p5
#define B p6
#define C p7

void
transform(char **s)
{
	int h;
	save();
	X = pop();
	F = pop();
	// save symbol context in case eval(B) below calls transform
	push(get_binding(symbol(METAA)));
	push(get_binding(symbol(METAB)));
	push(get_binding(symbol(METAX)));
	set_binding(symbol(METAX), X);
	// put constants in F(X) on the stack
	h = tos;
	push_integer(1);
	push(F);
	push(X);
	polyform(); // collect coefficients of x, x^2, etc.
	push(X);
	decomp_nib();
	while (*s) {
		scan_meta(*s);
		p1 = pop();
		A = cadr(p1);
		B = caddr(p1);
		C = cdddr(p1);
		if (f_equals_a(h))
			break;
		s++;
	}
	tos = h;
	if (*s) {
		push(B);
		eval();
		p1 = pop();
	} else
		p1 = symbol(NIL);
	set_binding(symbol(METAX), pop());
	set_binding(symbol(METAB), pop());
	set_binding(symbol(METAA), pop());
	push(p1);
	restore();
}

// search for a METAA and METAB such that F = A

int
f_equals_a(int h)
{
	int i, j;
	for (i = h; i < tos; i++) {
		set_binding(symbol(METAA), stack[i]);
		for (j = h; j < tos; j++) {
			set_binding(symbol(METAB), stack[j]);
			p1 = C;				// are conditions ok?
			while (iscons(p1)) {
				push(car(p1));
				eval();
				p2 = pop();
				if (iszero(p2))
					break;
				p1 = cdr(p1);
			}
			if (iscons(p1))			// no, try next j
				continue;
			push(F);			// F = A?
			push(A);
			eval();
			subtract();
			p1 = pop();
			if (iszero(p1))
				return 1;		// yes
		}
	}
	return 0;					// no
}

// Transpose tensor indices

void
eval_transpose(void)
{
	push(cadr(p1));
	eval();
	if (cddr(p1) == symbol(NIL)) {
		push_integer(1);
		push_integer(2);
	} else {
		push(caddr(p1));
		eval();
		push(cadddr(p1));
		eval();
	}
	transpose();
}

void
transpose(void)
{
	int i, j, k, l, m, ndim, nelem, t;
	int ai[MAXDIM], an[MAXDIM];
	U **a, **b;
	save();
	p3 = pop();
	p2 = pop();
	p1 = pop();
	if (!istensor(p1)) {
		if (!iszero(p1))
			stop("transpose: tensor expected, 1st arg is not a tensor");
		push(zero);
		restore();
		return;
	}
	ndim = p1->u.tensor->ndim;
	nelem = p1->u.tensor->nelem;
	// vector?
	if (ndim == 1) {
		push(p1);
		restore();
		return;
	}
	push(p2);
	l = pop_integer();
	push(p3);
	m = pop_integer();
	if (l < 1 || l > ndim || m < 1 || m > ndim)
		stop("transpose: index out of range");
	l--;
	m--;
	p2 = alloc_tensor(nelem);
	p2->u.tensor->ndim = ndim;
	for (i = 0; i < ndim; i++)
		p2->u.tensor->dim[i] = p1->u.tensor->dim[i];
	p2->u.tensor->dim[l] = p1->u.tensor->dim[m];
	p2->u.tensor->dim[m] = p1->u.tensor->dim[l];
	a = p1->u.tensor->elem;
	b = p2->u.tensor->elem;
	for (i = 0; i < ndim; i++) {
		ai[i] = 0;
		an[i] = p1->u.tensor->dim[i];
	}
	// copy components from a to b
	for (i = 0; i < nelem; i++) {
		t = ai[l]; ai[l] = ai[m]; ai[m] = t;
		t = an[l]; an[l] = an[m]; an[m] = t;
		k = 0;
		for (j = 0; j < ndim; j++)
			k = (k * an[j]) + ai[j];
		t = ai[l]; ai[l] = ai[m]; ai[m] = t;
		t = an[l]; an[l] = an[m]; an[m] = t;
		b[k] = a[i];
		for (j = ndim - 1; j >= 0; j--) {
			if (++ai[j] < an[j])
				break;
			ai[j] = 0;
		}
	}
	push(p2);
	restore();
}

// Evaluate a user defined function

#undef F
#undef A
#undef B
#undef S

#define F p3 // F is the function body
#define A p4 // A is the formal argument list
#define B p5 // B is the calling argument list
#define S p6 // S is the argument substitution list

int rewrite_args(void);
int rewrite_args_tensor(void);

void
eval_user_function(void)
{
	int h;
	// Use "derivative" instead of "d" if there is no user function "d"
	if (car(p1) == symbol(SYMBOL_D) && get_arglist(symbol(SYMBOL_D)) == symbol(NIL)) {
		eval_derivative();
		return;
	}
	F = get_binding(car(p1));
	A = get_arglist(car(p1));
	B = cdr(p1);
	// Undefined function?
	if (F == car(p1)) {
		h = tos;
		push(F);
		p1 = B;
		while (iscons(p1)) {
			push(car(p1));
			eval();
			p1 = cdr(p1);
		}
		list(tos - h);
		return;
	}
	// Create the argument substitution list S
	p1 = A;
	p2 = B;
	h = tos;
	while (iscons(p1) && iscons(p2)) {
		push(car(p1));
		push(car(p2));
		eval();
		p1 = cdr(p1);
		p2 = cdr(p2);
	}
	list(tos - h);
	S = pop();
	// Evaluate the function body
	push(F);
	if (iscons(S)) {
		push(S);
		rewrite_args();
	}
	eval();
}

// Rewrite by expanding symbols that contain args

int
rewrite_args(void)
{
	int h, n = 0;
	save();
	p2 = pop(); // subst. list
	p1 = pop(); // expr
	if (istensor(p1)) {
		n = rewrite_args_tensor();
		restore();
		return n;
	}
	if (iscons(p1)) {
		h = tos;
		push(car(p1)); // Do not rewrite function name
		p1 = cdr(p1);
		while (iscons(p1)) {
			push(car(p1));
			push(p2);
			n += rewrite_args();
			p1 = cdr(p1);
		}
		list(tos - h);
		restore();
		return n;
	}
	// If not a symbol then done
	if (!issymbol(p1)) {
		push(p1);
		restore();
		return 0;
	}
	// Try for an argument substitution first
	p3 = p2;
	while (iscons(p3)) {
		if (p1 == car(p3)) {
			push(cadr(p3));
			restore();
			return 1;
		}
		p3 = cddr(p3);
	}
	// Get the symbol's binding, try again
	p3 = get_binding(p1);
	push(p3);
	if (p1 != p3) {
		push(p2); // subst. list
		n = rewrite_args();
		if (n == 0) {
			pop();
			push(p1); // restore if not rewritten with arg
		}
	}
	restore();
	return n;
}

int
rewrite_args_tensor(void)
{
	int i, n = 0;
	push(p1);
	copy_tensor();
	p1 = pop();
	for (i = 0; i < p1->u.tensor->nelem; i++) {
		push(p1->u.tensor->elem[i]);
		push(p2);
		n += rewrite_args();
		p1->u.tensor->elem[i] = pop();
	}
	push(p1);
	return n;
}

//-----------------------------------------------------------------------------
//
//	Scan expr for vars, return in vector
//
//	Input:		Expression on stack
//
//	Output:		Vector
//
//-----------------------------------------------------------------------------

int global_h;

void
variables(void)
{
	int i, n;
	save();
	p1 = pop();
	global_h = tos;
	lscan(p1);
	n = tos - global_h;
	if (n > 1)
		qsort(stack + global_h, n, sizeof (U *), var_cmp);
	p1 = alloc_tensor(n);
	p1->u.tensor->ndim = 1;
	p1->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++)
		p1->u.tensor->elem[i] = stack[i];
	tos = global_h;
	push(p1);
	restore();
}

void
lscan(U *p)
{
	int i;
	if (iscons(p)) {
		p = cdr(p);
		while (iscons(p)) {
			lscan(car(p));
			p = cdr(p);
		}
	} else if (issymbol(p) && p != symbol(EXP1)) {
		for (i = global_h; i < tos; i++)
			if (stack[i] == p)
				return;
		push(p);
	}
}

int
var_cmp(const void *p1, const void *p2)
{
	return cmp_expr(*((U **) p1), *((U **) p2));
}

//-----------------------------------------------------------------------------
//
//	Encapsulate stack values in a vector
//
//	Input:		n		Number of values on stack
//
//			tos-n		Start of value
//
//	Output:		Vector on stack
//
//-----------------------------------------------------------------------------

void
vectorize(int n)
{
	int i;
	save();
	p1 = alloc_tensor(n);
	p1->u.tensor->ndim = 1;
	p1->u.tensor->dim[0] = n;
	for (i = 0; i < n; i++)
		p1->u.tensor->elem[i] = stack[tos - n + i];
	tos -= n;
	push(p1);
	restore();
}

void
eval_zero(void)
{
	int i, k[MAXDIM], m, n;
	m = 1;
	n = 0;
	p2 = cdr(p1);
	while (iscons(p2)) {
		push(car(p2));
		eval();
		i = pop_integer();
		if (i < 2) {
			push(zero);
			return;
		}
		m *= i;
		k[n++] = i;
		p2 = cdr(p2);
	}
	if (n == 0) {
		push(zero);
		return;
	}
	p1 = alloc_tensor(m);
	p1->u.tensor->ndim = n;
	for (i = 0; i < n; i++)
		p1->u.tensor->dim[i] = k[i];
	push(p1);
}
