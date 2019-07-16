#include <iostream>

#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <map>
#include <numeric>

#include "vw.h"
#include "rand48.h"
#include "conditional_contextual_bandit.h"

#include "common.h"
#include "diversity_experiments.h"
#include "ctr_experiments.h"
#include "discovery_experiments.h"

void print_usage(char* exe_name)
{
  std::cerr << "Usage: " << exe_name << " [1|2|3]\n"
            << "\t 1 - diversity\n"
            << "\t 2 - slot dependent ctr\n"
            << "\t 3 - discovery\n";
}

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    print_usage(argv[0]);
    return 1;
  }

  std::string arg_str(argv[1]);
  if (arg_str == "1")
  {
    diversity();
  }
  else if (arg_str == "2")
  {
    slot_dependent_ctr();
  }
  else if (arg_str == "3")
  {
    discovery_rate();
  }
  else
  {
    print_usage(argv[0]);
    return 1;
  }
}
