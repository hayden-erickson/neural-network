package main

import (
	"fmt"
	"strconv"

	"github.com/hayden-erickson/neural-network/loaders"
	"github.com/hayden-erickson/neural-network/nn"
)

func main() {
	// benchmarkParFor()
	runSGD()
}

func runSGD() {

	trainingData, testData := getData()

	inputSize, outputSize := len(testData[0].GetInput()), len(testData[0].GetOutput())

	net, e := nn.NewNetwork([]int{inputSize, outputSize})

	if e != nil {
		panic(e)
	}

	sgd := nn.SGD{
		Activation: nn.Sigmoid,
		Cost:       nn.CrossEntropy,
		Eta:        5,
		Net:        net,
	}

	fmt.Printf("# epoch\taccuracy/%d\tcost\n", len(testData))

	eFactor := 1

	numCorrect, cost := nn.Evaluate(sgd, testData, loaders.MnistMatcher)
	fmt.Printf("%d\t%d\t\t%f\n", 0, numCorrect, cost)

	// var oW []la.Matrix
	// var oB [][]float64

	for i := 0; i < 30; i++ {
		// copy(oW, sgd.Net.Weights)
		// copy(oB, sgd.Net.Biases)

		sgd.MRun(trainingData, eFactor, 10)
		// sgd.Run(trainingData[:N], eFactor, 10)

		// if reflect.DeepEqual(oW, sgd.Net.Weights) {
		// 	panic(`weights did not change`)
		// }

		numCorrect, cost := nn.Evaluate(sgd, testData, loaders.MnistMatcher)
		fmt.Printf("%d\t%d\t\t%f\n", eFactor*(i+1), numCorrect, cost)
	}
}

func filter(trainingData []nn.Example) []nn.Example {
	out := []nn.Example{}

	for _, e := range trainingData {
		output := e.GetOutput()
		if output[0] == 1 {
			out = append(out, e)
		}
		if output[1] == 1 {
			out = append(out, e)
		}
		// if output[2] == 1 {
		// 	out = append(out, e)
		// }
		// if output[3] == 1 {
		// 	out = append(out, e)
		// }
		// if output[4] == 1 {
		// 	out = append(out, e)
		// }
	}

	return out
}

func printRow(N int, sTime int64, pTimes []int64) {
	fmt.Printf("%d\t%d\t", N, sTime/1000)

	for i := 0; i < len(pTimes); i++ {
		fmt.Printf("%s\t", strconv.FormatInt(pTimes[i]/1000, 10))
	}

	fmt.Println()
}

func getData() (trainingData []nn.Example, testData []nn.Example) {
	var e error

	l := loaders.NewMNISTLoader(
		`./data/train-images-idx3-ubyte`,
		`./data/train-labels-idx1-ubyte`,
		2051,
		2049)

	trainingData, e = l.Load()
	// trainingData = filter(trainingData)

	if e != nil {
		panic(e)
	}

	l = loaders.NewMNISTLoader(
		`./data/t10k-images-idx3-ubyte`,
		`./data/t10k-labels-idx1-ubyte`,
		2051,
		2049)

	testData, e = l.Load()
	// testData = filter(testData)

	if e != nil {
		panic(e)
	}

	return trainingData, testData
}

// func Sigmoid(z float64) float64 {
// 	return 1.0 / (1.0 + math.Exp(-z))
// }

// func SigmoidPrime(z float64) float64 {
// 	return Sigmoid(z) * (1 - Sigmoid(z))
// }
