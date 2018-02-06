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

func (n Network) Prop(input []float64, activation Differentiable) []float64 {
	a := la.CreateVMapper(ToOP(activation.Fn))
	activations := input

	// sigmoid(wa + b)
	for i := 0; i < len(n.Weights); i++ {
		activations =
			a(getZ(activations, n.Biases[i], n.Weights[i]))
	}

	return activations
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

func (n Network) MBackProp(
	input la.Matrix,
	desired la.Matrix,
	a Differentiable,
	c Differentiable,
) (nablaW []la.Matrix, nablaB [][]float64) {

	nablaB = make([][]float64, len(n.Biases))
	nablaW = make([]la.Matrix, len(n.Weights))

	activations := []la.Matrix{input}
	var zs []la.Matrix

	for i := 0; i < len(n.Weights); i++ {
		z := la.MMapI(
			la.MMDot(n.Weights[i], activations[i]),
			la.MapVectorCol(n.Biases[i], la.SUM))

		zs = append(zs, z)
		activations = append(activations, la.MMap(z, ToOP(a.Fn)))
	}

	actual := activations[len(activations)-1]

	// delta := la.MULT(cPrime(actual, desired), aPrime(zs[len(zs)-1]))
	mCPrime := la.CreateMatrixOP(ToBOP(c.Prime))
	delta := la.MMULT(mCPrime(actual, desired), la.MMap(zs[len(zs)-1], ToOP(a.Prime)))

	nablaB[len(nablaB)-1] = rowReduceAvg(delta)
	nablaW[len(nablaW)-1] = mOuterColReduceAvg(delta, activations[len(activations)-2])

	numLayers := len(n.Weights) + 1

	for l := 2; l < numLayers; l++ {
		ap := la.MMap(zs[len(zs)-l], ToOP(a.Prime))
		w := n.Weights[(len(n.Weights)-l)+1]

		delta = la.MMULT(la.MMDot(w.T(), delta), ap)

		nablaB[len(nablaB)-l] = rowReduceAvg(delta)
		nablaW[len(nablaW)-l] = mOuterColReduceAvg(delta, activations[(numLayers-l)-1])
	}

	return nablaW, nablaB
}

func mOuterColReduceAvg(a, b la.Matrix) la.Matrix {
	out := make([]la.Matrix, a.Shape()[1])

	for j := 0; j < a.Shape()[1]; j++ {
		out[j] = la.Outer(a.Col(j), b.Col(j))
	}

	return la.MSCALE(la.MAddReduce(out), (1 / float64(a.Shape()[1])))

}

func rowReduceAvg(a la.Matrix) []float64 {
	out := make([]float64, a.Shape()[0])

	for i := 0; i < a.Shape()[0]; i++ {
		out[i] = la.AddReduce(a.Row(i))
	}

	return la.VSCALE(out, (1 / float64(a.Shape()[1])))
}

func NewNetwork(layers []int) (Network, error) {
	if len(layers) <= 1 {
		return Network{}, errors.New(`Cannot initialize network with less than 2 layers`)
	}

	weights := make([]la.Matrix, len(layers)-1)
	biases := make([][]float64, len(layers)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = la.RandMatrix(layers[i+1], layers[i])
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
