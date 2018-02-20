package nn

import (
	"errors"

	"github.com/hayden-erickson/neural-network/la"
)

type Example interface {
	GetInput() []float64
	GetOutput() []float64
}

type Network struct {
	Weights []la.Matrix
	Biases  [][]float64
}

func getZ(a, b []float64, w la.Matrix) []float64 {
	return la.VSUM(la.MVDot(w, a), b)
}

func (n Network) Prop(input []float64, aFunc Differentiable) []float64 {
	a := la.CreateVMapper(ToOP(aFunc.Fn))
	activation := input

	// sigmoid(wa + b)
	for i := 0; i < len(n.Weights); i++ {
		activation =
			a(la.VSUM(la.MVDot(n.Weights[i], activation), n.Biases[i]))
	}

	return activation
}

func (n Network) BackProp(
	e Example,
	activation Differentiable,
	cost Differentiable,
) ([]la.Matrix, [][]float64) {
	nablaB := make([][]float64, len(n.Biases))
	nablaW := make([]la.Matrix, len(n.Weights))
	a := la.CreateVMapper(ToOP(activation.Fn))
	aPrime := la.CreateVMapper(ToOP(activation.Prime))
	cPrime := la.CreateVectorOP(ToBOP(cost.Prime))

	activations := [][]float64{e.GetInput()}
	var zs [][]float64

	for i := 0; i < len(n.Weights); i++ {
		z := getZ(activations[i], n.Biases[i], n.Weights[i])
		zs = append(zs, z)
		activations = append(activations, a(z))
	}

	actual := activations[len(activations)-1]
	desired := e.GetOutput()

	delta := la.VMULT(cPrime(actual, desired), aPrime(zs[len(zs)-1]))

	nablaB[len(nablaB)-1] = delta
	nablaW[len(nablaW)-1] = la.Outer(delta, activations[len(activations)-2])

	numLayers := len(n.Weights) + 1

	for l := 2; l < numLayers; l++ {
		ap := aPrime(zs[len(zs)-l])
		w := n.Weights[(len(n.Weights)-l)+1]

		delta = la.VMULT(la.MVDot(w.T(), delta), ap)

		nablaB[len(nablaB)-l] = delta
		nablaW[len(nablaW)-l] = la.Outer(delta, activations[(numLayers-l)-1])
	}

	return nablaW, nablaB
}

func (n Network) Saturate(input la.Matrix, a Differentiable) (weighted, activations []la.Matrix) {
	activations = []la.Matrix{input}

	// === Propagate forward ===
	// add activation and z
	for i := 0; i < len(n.Weights); i++ {
		z := la.MMapI(
			la.MMDot(n.Weights[i], activations[i]),
			la.MapVectorCol(n.Biases[i], la.SUM))

		weighted = append(weighted, z)
		activations = append(activations, la.MMapD(z, ToOP(a.Fn)))
	}

	return weighted, activations
}

func Delta(actual, desired, weighted la.Matrix, a, c Differentiable) la.Matrix {
	mCPrime := la.CreateMatrixOP(ToBOP(c.Prime))
	// Quadratic
	// return la.MMULT(mCPrime(actual, desired), la.MMapD(weighted, ToOP(a.Prime)))
	// Cross Entropy
	return mCPrime(actual, desired)
}

func (n Network) MBackProp(
	input la.Matrix,
	desired la.Matrix,
	a Differentiable,
	c Differentiable,
) (nablaW []la.Matrix, nablaB [][]float64) {

	nablaB = make([][]float64, len(n.Biases))
	nablaW = make([]la.Matrix, len(n.Weights))

	// === Propagate forward ===
	zs, activations := n.Saturate(input, a)

	actual := activations[len(activations)-1]

	// === Compute Delta ===
	// delta := la.MULT(cPrime(actual, desired), aPrime(zs[len(zs)-1]))
	// mCPrime := la.CreateMatrixOP(ToBOP(c.Prime))
	// delta := la.MMULT(mCPrime(actual, desired), la.MMapD(zs[len(zs)-1], ToOP(a.Prime)))
	delta := Delta(actual, desired, zs[len(zs)-1], a, c)

	nablaB[len(nablaB)-1] = la.RowAvg(delta)
	nablaW[len(nablaW)-1] = la.MOuterColAvg(delta, activations[len(activations)-2])

	numLayers := len(n.Weights) + 1

	// === Back Propagate ===
	for l := 2; l < numLayers; l++ {
		ap := la.MMap(zs[len(zs)-l], ToOP(a.Prime))
		w := n.Weights[(len(n.Weights)-l)+1]

		delta = la.MMULT(la.MMDot(w.T(), delta), ap)

		nablaB[len(nablaB)-l] = la.RowAvg(delta)
		nablaW[len(nablaW)-l] = la.MOuterColAvg(delta, activations[(numLayers-l)-1])
	}

	return nablaW, nablaB
}

func NewNetwork(layers []int) (Network, error) {
	if len(layers) <= 1 {
		return Network{}, errors.New(`Cannot initialize network with less than 2 layers`)
	}

	weights := make([]la.Matrix, len(layers)-1)
	biases := make([][]float64, len(layers)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = la.RandMatrixSquashed(layers[i+1], layers[i])
		biases[i] = la.RandVector(layers[i+1])
	}

	return Network{
		Weights: weights,
		Biases:  biases,
	}, nil
}

func CopyNetwork(into, from *Network) {
	into.Weights = make([]la.Matrix, len(from.Weights))
	into.Biases = make([][]float64, len(from.Biases))

	for i, _ := range from.Weights {
		into.Weights[i] = la.ZeroMatrix(from.Weights[i].Shape()[0], from.Weights[i].Shape()[1])
		into.Biases[i] = make([]float64, len(from.Biases[i]))

		into.Weights[i] = from.Weights[i]
		copy(into.Biases[i], from.Biases[i])
	}
}
