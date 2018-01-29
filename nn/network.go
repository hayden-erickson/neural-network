package nn

import (
	"errors"
	"fmt"

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
	return la.SUM(la.MVDot(w, a), b)
}

func (n Network) Prop(input []float64, activation la.Mapper) []float64 {
	activations := input

	// sigmoid(wa + b)
	for i := 0; i < len(n.Weights); i++ {
		activations =
			activation(getZ(activations, n.Biases[i], n.Weights[i]))
	}

	return activations
}

func (n Network) BackProp(
	e Example,
	a la.Mapper,
	aPrime la.Mapper,
	cPrime la.Aggregator,
) ([]la.Matrix, [][]float64) {
	nablaB := make([][]float64, len(n.Biases))
	nablaW := make([]la.Matrix, len(n.Weights))

	activations := [][]float64{e.GetInput()}
	var zs [][]float64

	for i := 0; i < len(n.Weights); i++ {
		z := getZ(activations[i], n.Biases[i], n.Weights[i])
		zs = append(zs, z)
		activations = append(activations, a(z))
	}

	actual := activations[len(activations)-1]
	desired := e.GetOutput()

	delta := la.MULT(cPrime(actual, desired), aPrime(zs[len(zs)-1]))

	nablaB[len(nablaB)-1] = delta
	nablaW[len(nablaW)-1] = la.Outer(delta, activations[len(activations)-2])

	numLayers := len(n.Weights) + 1

	for l := 2; l < numLayers; l++ {
		ap := aPrime(zs[len(zs)-l])
		w := n.Weights[(len(n.Weights)-l)+1]

		delta = la.MULT(la.MVDot(w.T(), delta), ap)

		nablaB[len(nablaB)-l] = delta
		nablaW[len(nablaW)-l] = la.Outer(delta, activations[(numLayers-l)-1])
	}

	return nablaW, nablaB
}

func (n Network) MBackProp(
	input la.Matrix,
	desired la.Matrix,
	a la.OP,
	aPrime la.OP,
	cPrime la.BOP,
) (nablaW []la.Matrix, nablaB []la.Matrix) {

	nablaB = make([]la.Matrix, len(n.Biases))
	nablaW = make([]la.Matrix, len(n.Weights))

	activations := []la.Matrix{input}
	var zs []la.Matrix

	for i := 0; i < len(n.Weights); i++ {
		// z := getZ(activations[i], n.Biases[i], n.Weights[i])
		zi := la.MMDot(n.Weights[i], activations[i])
		z := la.Matrix{X: len(n.Biases[i]), Y: activations[i].Y}

		for i, b := range n.Biases[i] {
			z.Data = append(z.Data, la.Map(zi.Row(i), la.Add(b))...)
		}

		zs = append(zs, z)
		activations = append(activations, la.MMap(z, a))
	}

	actual := activations[len(activations)-1]

	// delta := la.MULT(cPrime(actual, desired), aPrime(zs[len(zs)-1]))
	mCPrime := la.CreateMatrixAgg(cPrime)
	delta := la.MMULT(mCPrime(actual, desired), la.MMap(zs[len(zs)-1], aPrime))

	fmt.Println(delta.X, delta.Y)

	// deltaV := make([]float64, delta.X)

	// for i := 0; i < delta.X; i++ {
	// 	deltaV[i] = la.AddReduce(delta.Row(i))
	// }

	// // TODO: take avg over all input examples
	// deltaV = la.Map(deltaV, la.MultBy(1/float64(len(deltaV))))

	// nablaB[len(nablaB)-1] = delta
	// nablaW[len(nablaW)-1] = la.Outer(delta, activations[len(activations)-2])

	// numLayers := len(n.Weights) + 1

	// for l := 2; l < numLayers; l++ {
	// 	ap := aPrime(zs[len(zs)-l])
	// 	w := n.Weights[(len(n.Weights)-l)+1]

	// 	delta = la.MULT(la.MVDot(w.T(), delta), ap)

	// 	nablaB[len(nablaB)-l] = delta
	// 	nablaW[len(nablaW)-l] = la.Outer(delta, activations[(numLayers-l)-1])
	// }

	return nablaW, nablaB
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
		into.Weights[i] = la.ZeroMatrix(from.Weights[i].X, from.Weights[i].Y)
		into.Biases[i] = make([]float64, len(from.Biases[i]))

		la.CopyMatrix(into.Weights[i], from.Weights[i])
		copy(into.Biases[i], from.Biases[i])
	}
}
