#include <cmath>

#include "config.hpp"
#include "activations.hpp"

namespace activation
{
	nntype ReLU(nntype x) { return x < 0 ? 0 : x; }
	nntype leakyReLU(nntype x) { return x < 0 ? x * 0.001 : x; }
	nntype sigmoid(nntype x) { return 1.0f / (1.0f + std::exp(-x)); }

	nntype dReLU(nntype x) { return x; }
	nntype dLeakyReLU(nntype x) { return x < 0 ? x * 1000 : x; }
	nntype dSigmoid(nntype x) { return x * (1 - x); }
}