forked from [georgeweigt/eigenmath](https://github.com/georgeweigt/eigenmath)

The Eigenmath manual and sample scripts are available at https://georgeweigt.github.io

To build and run:

	cd src
	make
	./eigenmath

Press ctrl-C to exit.

To run a script:

	./eigenmath scriptfilename

To generate LaTeX output:

	./eigenmath --latex scriptfilename | tee foo.tex

To generate MathML output:

	./eigenmath --mathml scriptfilename | tee foo.html

To generate MathJax output:

	./eigenmath --mathjax scriptfilename | tee foo.html

Eigenmath uses UTF-8 encoding to display Greek letters and large delimiters.

	      ┌                                  ┐
	      │ −ξ(r)     0      0        0      │
	      │                                  │
	      │           1                      │
	      │   0     ╶────╴   0        0      │
	      │          ξ(r)                    │
	g   = │                                  │
	 μν   │                   2              │
	      │   0       0      r        0      │
	      │                                  │
	      │                        2       2 │
	      │   0       0      0    r  sin(θ)  │
	      └                                  ┘

