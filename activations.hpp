#include "config.hpp"

namespace activation
{
	nntype ReLU(nntype x);
	nntype leakyReLU(nntype x);
	nntype sigmoid(nntype x);

	nntype dReLU(nntype x);
	nntype dLeakyReLU(nntype x);
	nntype dSigmoid(nntype x);
}