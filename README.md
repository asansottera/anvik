anvik
=====

Anvik computes the optimal allocation policy for virtual machines.
This policy can later be used by an online allocator, such as the ones used in Infrastructure-as-a-Service clouds.

The problem is formulated as a Markov Decision Process.
Different classes of virtual machines and heterogeneous servers are considered.
Virtual machine arrivals are assumed to follow a Poisson process.

Two implementations of the solution algorithm are provided.
The first version runs on CPUs, is written in C++ and uses OpenMP to exploit shared-memory parallel architectures.
The second version runs on GPUs and is written in CUDA. 
A lot of optimization effort was posed on reducing the amount of memory required.

The command line version of the tool has been tested on both Linux and Windows.
A graphical user interface, written in C# and C++/CLI is provided for Windows machines.

Anvik is a joint work of <a href="http://home.deib.polimi.it/sansottera/">Andrea Sansottera</a> and <a href="http://home.deib.polimi.it/cremones/">Paolo Cremonesi</a>.
If you use Anvik for your research, please cite the following paper, which will be presented at ValueTools 2013:
