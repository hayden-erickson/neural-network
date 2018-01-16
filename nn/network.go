package nn

import (
	"errors"
	"math"

	"github.com/hayden-erickson/neural-network/la"
)

type Network struct {
	Weights   []la.Matrix
	Biases    []la.Vector
	NumLayers int
}

type Example interface {
	GetInput() la.Vector
	GetOutput() la.Vector
}

func (n Network) Passthrough(la.Vector) la.Vector {
	return []la.Scalar{}
}

func (n Network) Train(testData []Example, epochs int, miniBatchSize int, eta float64) {
}

func NewNetwork(layers []int) (Network, error) {
	if len(layers) <= 1 {
		return Network{}, errors.New(`Cannot initialize network with less than 2 layers`)
	}

	weights := make([]la.Matrix, len(layers)-1)
	biases := make([]la.Vector, len(layers)-1)

	for i := 0; i < len(weights); i++ {
		weights[i] = la.RandMatrix(layers[i+1], layers[i])
		biases[i] = la.RandVector(layers[i+1])
	}

	return Network{
		NumLayers: len(layers),
		Weights:   weights,
		Biases:    biases,
	}, nil
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func sigmoidPrime(z float64) float64 {
	return sigmoid(z) * (1 - sigmoid(z))
}
