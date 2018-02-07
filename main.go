package main

import (
	"fmt"
	"strconv"
	"time"

	"github.com/hayden-erickson/neural-network/la"
	"github.com/hayden-erickson/neural-network/loaders"
	"github.com/hayden-erickson/neural-network/nn"
	"github.com/hayden-erickson/neural-network/parallel"
)

func main() {
	benchmarkParFor()
}

type tester struct {
	data []float64
	op   la.OP
}

func (t tester) getSerialTime() int64 {
	out := make([]float64, len(t.data))

	start := time.Now().UnixNano()

	for i := 0; i < len(t.data); i++ {
		out[i] = t.op(t.data[i])
	}

	finish := time.Now().UnixNano()

	return finish - start
}

func (t tester) getParallelTime(branch int) int64 {

	start := time.Now().UnixNano()

	parallel.For(t.data, branch, t.op)

	finish := time.Now().UnixNano()

	return finish - start
}

func printRow(N int, sTime int64, pTimes []int64) {
	fmt.Printf("%d\t%d\t", N, sTime/1000)

	for i := 0; i < len(pTimes); i++ {
		fmt.Printf("%s\t", strconv.FormatInt(pTimes[i]/1000, 10))
	}

	fmt.Println()
}

func benchmarkParFor() {
	var N int
	var i uint

	fmt.Println("# N\t0\t2\t4\t8\t16")
	pTime := make([]int64, 4)

	for i = 10; i < 28; i++ {
		N = 1 << i

		t := tester{data: la.RandVector(N), op: la.MultBy(5.0)}
		sTime := t.getSerialTime()
		pTime[0] = t.getParallelTime(1)
		pTime[1] = t.getParallelTime(2)
		pTime[2] = t.getParallelTime(3)
		pTime[3] = t.getParallelTime(4)

		printRow(N, sTime, pTime)
	}
}

func runSGD() {
	net, e := nn.NewNetwork([]int{784, 30, 10})

	if e != nil {
		panic(e)
	}

	sgd := nn.SGD{
		Activation: nn.Sigmoid,
		Cost:       nn.Quadratic,
		Eta:        5,
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

	fmt.Printf("epoch\taccuracy\ttime\n")

	for i := 0; i < 30; i++ {
		sgd.MRun(trainingData, 1, 10)
		fmt.Printf("%d\t%f\t%d\n", i+1, float64(Evaluate(sgd.Net, testData))/100.0, time.Now().UnixNano())
	}
}

func Evaluate(n nn.Network, testData []nn.Example) int {
	var correct int

	for _, e := range testData {
		if match(n.Prop(e.GetInput(), nn.Sigmoid), e.GetOutput()) {
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
