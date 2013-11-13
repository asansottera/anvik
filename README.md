Anvik
=====

Anvik computes the optimal allocation policy for virtual machines.
This policy can later be used by an online allocator in Infrastructure-as-a-Service clouds.

The problem is formulated as a Markov Decision Process.
Different classes of virtual machines and heterogeneous servers are considered.
Virtual machine arrivals are assumed to follow a Poisson process.
Revenue rates and cost rates are associated to virtual machines and severs, respectively.
The user can decides whether virtual machines can be rejected or not.
If this option is enabled, a joint allocation and admission control problem is solved.

Two implementations of the solution algorithm are provided.
The first version runs on CPUs, is written in C++ and uses OpenMP to exploits shared-memory parallel architectures.
The second version runs on GPUs and is written in CUDA. 
The GPU code has both a compute-optimized path and a memory-optimized path, and
automatically selects one of the two depending on the problem size and the available GPU memory.

The command line version of the tool has been tested on both Linux and Windows machines.
A graphical user interface, written in C# and C++/CLI, is provided for Windows machines.

Anvik is a joint work of <a href="http://home.deib.polimi.it/sansottera/">Andrea Sansottera</a> and <a href="http://home.deib.polimi.it/cremones/">Paolo Cremonesi</a>.
If you use Anvik for your research, please cite the following paper, which will be presented at ValueTools 2013:

Andrea Sansottera, Paolo Cremonesi,
*Optimal Virtual Machine Allocation with Anvik*,
[To appear] Conference on Performance Evaluation Methodologies and Tools (ValueTools), 2013
<a href="http://home.deib.polimi.it/sansottera/bibtex/sansottera2013valuetools.bib">Bibtex</a>
