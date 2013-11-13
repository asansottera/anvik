#ifndef ANVIK_EXAMPLES_H
#define ANVIK_EXAMPLES_H

#include <cstdint>
#include <vector>
#include "problem.h"

/* Creates an example problem with four types of server and three types of VM. */
problem example01();

/* Creates an example problem with a single type of server and three types of VM. */
problem example02();

/* Creates an example problem with a single type of server and a single type of VM.
   The number of servers is specified as an argument. */
problem example03(uint32_t N);

/* Creates an example problem with a single type of server and two types of VM. */
problem example04();

/* Creates an example problem with two types of server and two types of VM.
   The number of the servers of each type are specified as arguments. */
problem example05(uint32_t n1, uint32_t n2);

/* Creates a trivial example problem with one type of server and one types of VM.
   Each server can host one VM.
   The number of servers is specified as an argument. */
problem example06(uint32_t n0);

/* Creates a trivial example problem with one type of server and one types of VM.
   Each server can host two VMs.
   The number of servers is specified as an argument. */
problem example07(uint32_t n0);

#endif