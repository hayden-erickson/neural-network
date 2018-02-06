package main

import (
	"fmt"
	"time"

	"github.com/hayden-erickson/neural-network/loaders"
	"github.com/hayden-erickson/neural-network/nn"
)

func main() {
	net, e := nn.NewNetwork([]int{784, 30, 10})

	if e != nil {
		panic(e)
	}

	sgd := nn.SGD{
		Activation: nn.Sigmoid,
		Cost:       nn.Quadratic,
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

	fmt.Printf("accuracy: %d/%d\n", Evaluate(sgd.Net, testData), len(testData))

	sgd.Run(trainingData[0:100], 5, 10)

	fmt.Printf("accuracy: %d/%d\n", Evaluate(sgd.Net, testData), len(testData))

	sgd.MRun(trainingData[0:100], 5, 10)

	fmt.Printf("M accuracy: %d/%d\n", Evaluate(sgd.Net, testData), len(testData))

	fmt.Printf("Done running SGD %s\n", time.Now().Truncate(time.Second))

}

func Evaluate(n nn.Network, testData []nn.Example) int {
	var correct int

	for _, e := range testData {
		if match(n.Prop(e.GetInput(), nn.Sigmoid{}), e.GetOutput()) {
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
