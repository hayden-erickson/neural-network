package main

import (
	"fmt"
	"math"
	"time"

	"github.com/hayden-erickson/neural-network/la"
	"github.com/hayden-erickson/neural-network/loaders"
	"github.com/hayden-erickson/neural-network/nn"
)

func main() {
	net, e := nn.NewNetwork([]int{784, 30, 10})

	if e != nil {
		panic(e)
	}

	sgd := nn.SGD{
		Activation: sigmoid,
		APrime:     sigmoidPrime,
		CostPrime:  costPrime,
		Eta:        3,
		Net:        net,
	}

	l := loaders.NewMNISTLoader(
		`./data/train-images-idx3-ubyte`,
		`./data/train-labels-idx1-ubyte`,
		2051,
		2049)

	trainingData, e := l.Load()

	if e != nil {
		panic(e)
	}

	epochs := 30
	var new nn.Network

	l = loaders.NewMNISTLoader(
		`./data/t10k-images-idx3-ubyte`,
		`./data/t10k-labels-idx1-ubyte`,
		2051,
		2049)

	testData, e := l.Load()

	if e != nil {
		panic(e)
	}

	fmt.Printf("Running SGD... %s\n", time.Now().Truncate(time.Second))

	for i := 0; i < epochs; i++ {
		new = sgd.Run(trainingData, 1, 10)
		sgd.Net = new

		if i%2 == 0 {
			fmt.Printf("accuracy: %d/%d\n", evaluate(new, testData), len(testData))
		}
	}

	fmt.Printf("Done running SGD %s\n", time.Now().Truncate(time.Second))

}

func evaluate(n nn.Network, testData []nn.Example) int {
	var correct int

	for _, e := range testData {
		if match(n.Prop(e.GetInput(), sigmoid), e.GetOutput()) {
			correct++
		}
	}

	return correct
}

func match(a, b []float64) bool {
	aMax := a[0]
	bMax := b[0]
	var aIdx, bIdx int

	for i, _ := range a {
		if a[i] > aMax {
			aMax = a[i]
			aIdx = i
		}

		if b[i] > bMax {
			bMax = b[i]
			bIdx = i
		}
	}

	return bIdx == aIdx
}

// func Sigmoid(z float64) float64 {
// 	return 1.0 / (1.0 + math.Exp(-z))
// }

// func SigmoidPrime(z float64) float64 {
// 	return Sigmoid(z) * (1 - Sigmoid(z))
// }

func sigmoid(z []float64) []float64 {
	act := func(z float64) float64 {
		return 1 / (1 + math.Exp(-z))
	}

	return la.Map(z, act)
}

func sigmoidPrime(z []float64) []float64 {
	oneMinus := func(x float64) float64 {
		return 1 - x
	}

	return la.MULT(sigmoid(z), la.Map(sigmoid(z), oneMinus))
}

func cost(desired, actual [][]float64) float64 {

	intermediate := make([]float64, len(desired))

	for i, _ := range desired {
		yMinA := la.SUB(desired[i], actual[i])
		intermediate[i] = la.AddReduce(la.MULT(yMinA, yMinA))
	}

	return la.AddReduce(intermediate) / float64(2*len(desired))
}

func costPrime(actual, desired []float64) []float64 {
	return la.SUB(actual, desired)
}
