# Random Kitchen Sinks

This library will eventually implement variants of the random kitchen sinks feature generation algorithm for nonlinear modeling. It will depend on the [Armadilllo](http://arma.sourceforge.net/) linear algebra library. I may eventually wish to extend an interface to python (numpy).

my goal is to implement:

* Random Fourier Features (Gaussian)
* Fastfood features (Gaussian, Polynomial)
* Orthogonal Random Features (Gaussian)
* Structures Orthogonal Random Features (Gaussian)
* ... and whatever I may come accross in the literature.

the following references should be of interest:

Rahimi, Ali and Benjamin Recht (2008). “Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning”. In: Advances in neural information processing systems, pp. 1313–1320.

^ this is the third paper in a trilogy


Felix, X Yu et al. (2016). “Orthogonal random features”. In: Advances in Neural Information Processing Systems, pp. 1975–1983.

Quoc Le, Tam ́as Sarl ́os and Alex Smola (2013).
“Fastfood-approximating kernel expansions in loglinear time”. In: Proceedings of the international conference on machine learning vol. 85.

Halko, Nathan, Per-Gunnar Martinsson, and Joel A. Tropp (2009). “Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions”. In: SIAM vol. 53, pp. 217–288.